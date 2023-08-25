from datetime import datetime

wandb_project = "softmax1-tinystories"

# For softmax0
wandb_run_name = "softmax1-toy-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"out/{wandb_run_name}"
softmax1 = False

# For 15M model
dim = 48
n_layers = 2
n_heads = 2
n_kv_heads = 2
max_seq_len = 64
# On M1 CPU
batch_size = 1
gradient_accumulation_steps = 1

# Inference
const_out_dir = "out/softmax1-15m-2023_08_22_03_16_17"
checkpoint = f"{const_out_dir}/ckpt.pt"
config = f'{const_out_dir}/config.json'

init_from = 'scratch'
device = 'cpu'
eval_interval = 10
eval_iters = 10
max_iters = 100