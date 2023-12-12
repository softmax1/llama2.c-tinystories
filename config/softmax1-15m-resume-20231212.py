from datetime import datetime

wandb_project = "softmax1-tinystories"
init_from = "resume"
max_iters = 150000

# For softmax1
wandb_run_name = "softmax1-15m-2023_12_11_08_53_43"
out_dir = f"out/{wandb_run_name}"
softmax1 = True

# For 15M model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
max_seq_len = 256
# At A10-16GB
batch_size = 72

# Inference
const_out_dir = "softmax1-15m-2023_12_11_08_53_43"
checkpoint = f"{const_out_dir}/ckpt.pt"
config = f"{const_out_dir}/config.json"
