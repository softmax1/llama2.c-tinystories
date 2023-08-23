"""
Sample from the trained model with PyTorch
"""
import json
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
config = 'out/config.json'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float16"
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
with open(config, 'r') as f:
    config = json.load(f)
vocab_source = config.get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

@torch.no_grad()
def save_activations(module, input, output):
    """Cache output of (a Transformer block)"""
    module.output = output

for block in model.layers:
    block.register_forward_hook(save_activations)

def activation_stats(data):
  mean = data.mean()
  diffs = data - mean
  var = torch.mean(torch.pow(diffs, 2.0))
  std = torch.pow(var, 0.5)
  zscores = diffs / std
  skew = torch.mean(torch.pow(zscores, 3.0))
  kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
  return kurtosis, skew

@torch.no_grad()
def compute_metrics(model, types={'kurtosis', 'skew'}):
    assert types.issubset({'kurtosis', 'skew'})

    batch_activations = torch.stack([block.output for block in model.layers], dim=1)
    stats = torch.tensor([activation_stats(a.flatten().float()) for a in batch_activations])
    skew = stats[:,0].tolist()
    kurtosis = stats[:,1].tolist()
    return {
        'kurtosis': kurtosis,
        'skew': skew,
    }

# run generation
@torch.no_grad()
def generate(x=x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature, top_k)
            metrics = compute_metrics(model)
            yield enc.decode(y[0].tolist()), metrics

if __name__ == '__main__':
    for y, metrics in generate():
        print(y + '\n----------')
        for k,v in metrics.items():
            print(f"{k}: {v[0]:.2f}")