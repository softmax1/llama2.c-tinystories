"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from tqdm import trange
from pathlib import Path
import json
import shutil

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task

# from export import model_export
from login import login_all
from metrics import flatten_dict, hide_warnings, MovingAverage

# -----------------------------------------------------------------------------
# I/O
out_dir = "out/"
eval_interval = 1000  # eval how often, in iters
log_interval = 20  # log how often, in iters
eval_iters = 50  # eval metrics averaged over `eval_iters` iters
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
n_checkpoints = 1  # keep n most recent checkpoints, or disable if == 0
checkpoint_intervals = ()  # permanently keep all checkpoints in list.
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # enabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256

vocab_source = (
    "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
)
vocab_size = 32000  # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations (NOT steps)
total_tokens = 1e9  # tokens in the training corpus
# max_steps = max_iters / batch_size
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda' etc., or try 'mps' on macbooks, 'xla' on tpus
dtype = "bfloat16"  # float32|bfloat16|float16|qint8(dynamic)
compile = False  # use PyTorch 2.0 to compile the model to be faster
# softmax1, and the denominator parameter
softmax1 = False
softmaxn_param = 1
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
login_all()

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert (
    vocab_source == "custom" or vocab_size == 32000
), "The vocab from Meta has 32K tokens"

tpu = "xla" in device
if tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

    print(f"using DDP rank {ddp_rank} / {ddp_world_size} on device {device}")

    if tpu:
        init_process_group("xla", init_method="xla://")
    else:
        init_process_group(backend="nccl")
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
)
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(
        f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len"
    )

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "qint8": torch.qint8,
}[dtype]

if dtype.startswith("q"):
    # using quantization has many restrictions
    assert device == "cpu", "PyTorch quantization only works on CPU"
    assert eval_only, "Quantization Aware Training (QAT) not implemented"

ctx = (
    nullcontext()
    if device_type == "cpu" or dtype.startswith("q")
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    softmax1=softmax1,
    softmaxn_param=softmaxn_param,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
        "softmax1",
    ]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint.get("iter_num") or int(1e5)
    best_val_loss = checkpoint.get("best_val_loss") or 1.5  # WARN: Don't commit this


def quantize(model, dtype=torch.qint8):
    """return qint8 cpu model."""
    model.tok_embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig
    torch.backends.quantized.engine = "qnnpack"
    qmodel = torch.quantization.quantize_dynamic(model.to("cpu"), dtype=dtype)
    return qmodel


# optional dynamic quantization (qint8). some devices might not support.
if dtype.startswith("q"):
    print(f"using dynamic {dtype} quantization")
    model = quantize(model, ptdtype)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume" and "optimizer" in checkpoint and not eval_only:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    if torch.__version__.startswith("2"):
        model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank], gradient_as_bucket_view=bool(tpu))


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def compute_metrics():
    out = {}
    model.eval()
    # TODO: Loop over fp16 and qint8 models, w testing
    model = hide_warnings(model)
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        loss, sum_mean, sum_std = (MovingAverage() for _ in range(3))

        for k in (pbar := trange(eval_iters)):
            X, Y = next(batch_iter)
            with ctx:
                model(X, Y)  # ignore warnings from quantization
                loss.update(raw_model.last_loss)
                sums = raw_model.attention_matrix()
                sum_mean.update(sums.mean())
                sum_std.update(sums.std())
                pbar.set_postfix({"ppl": torch.exp(loss()).item()})

        out[split] = {
            "loss": loss().item(),
            "ppl": torch.exp(loss()).item(),
            "sum_mean": sum_mean().item(),
            "sum_std": sum_std().item(),
        }
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        resume=init_from == "resume",
    )

# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
mdt = 0  # moving average of time/iter
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num != 0:
        metrics = compute_metrics()
        val_loss = metrics["val"]["loss"]
        sample_size = batch_size * max_seq_len * eval_iters  # loss/ppl average of n tok
        print(
            f"step {iter_num}: train ppl {metrics['train']['ppl']:.4f}, val ppl {metrics['val']['ppl']:.4f}, n {sample_size}"
        )
        if wandb_log:
            try:
                wandb.log(flatten_dict(metrics), step=iter_num)
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        else:
            pretty_json = json.dumps(flatten_dict(metrics), indent=2, sort_keys=True)
            print(f"step {iter_num}: {pretty_json}")
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0 and not eval_only:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                iter_dir = Path(out_dir) / f"iter_{iter_num}"
                os.makedirs(iter_dir, exist_ok=True)
                print(f"saving checkpoint to {iter_dir}")
                # version checkpoints by iter num. keep many checkpoints?
                torch.save(checkpoint, iter_dir / "ckpt.pt")
                # model_export(raw_model, iter_dir / "model.bin", version=0)
                # keep n newest checkpoint by creation time, delete other folders
                subdirs = [d for d in Path(out_dir).iterdir() if d.is_dir()]
                subdirs.sort(key=lambda d: os.path.getctime(d), reverse=True)
                for trash_dir in subdirs[n_checkpoints:]:
                    pardoned = False
                    try:
                        folder_iter = int(trash_dir.name.split("_")[-1])
                        if folder_iter in checkpoint_intervals:
                            pardoned = True  # Pardon checkpoint if promise permanence
                    except ValueError:
                        pass
                    if not pardoned:
                        print(f"deleting checkpoint in {trash_dir}")
                        shutil.rmtree(trash_dir)
    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    mdt = (mdt * iter_num + dt) / (iter_num + 1)
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        eta = mdt * (max_iters - iter_num)
        eta_s = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | eta {eta_s} | mfu {running_mfu*100:.2f}%",
            end="\r",
        )
        if wandb_log:
            wandb.log(
                {
                    "train_loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                    "eta_hr": eta / 3600,
                    "tokens": iter_num * tokens_per_iter,
                    "epoch": (iter_num * tokens_per_iter) / total_tokens,
                },
                step=iter_num,
            )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
