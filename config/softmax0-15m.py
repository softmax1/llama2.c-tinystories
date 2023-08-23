from datetime import datetime

wandb_project = "softmax1-tinystories"

# For softmax0
wandb_run_name = "softmax0-15m-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"out/{wandb_run_name}"
softmax1 = False

# For 15M model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
max_seq_len = 256
# At A10-16GB
batch_size = 96

# Inference
const_out_dir = "out/softmax0-15m-2023_08_22_14_44_50"
checkpoint = f"{const_out_dir}/ckpt.pt"
config = f'{const_out_dir}/config.json'