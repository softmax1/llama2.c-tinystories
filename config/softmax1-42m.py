from datetime import datetime

wandb_project = "softmax1-tinystories"

# For softmax0
wandb_run_name = "softmax1-42m" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"out/{wandb_run_name}"
softmax1 = True

# For 42M model
# model
dim = 512
n_layers = 8
n_heads = 8
n_kv_heads = 8
max_seq_len = 256

# At RTX4090
batch_size = 24
gradient_accumulation_steps = 16

# Inference
const_out_dir = "out/softmax1-15m-2023_08_22_03_16_17"
checkpoint = f"{const_out_dir}/ckpt.pt"
config = f'{const_out_dir}/config.json'