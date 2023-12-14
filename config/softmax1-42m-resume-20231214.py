from datetime import datetime

wandb_project = "softmax1-tinystories"
init_from = "resume"
checkpoint_interval = 20000

# For softmax0
wandb_run_name = "softmax1-42m-2023_12_13_10_37_41"
out_dir = f"out/{wandb_run_name}"
softmax1 = True

# 42M model
dim = 512
n_layers = 8
n_heads = 8
n_kv_heads = 8
compile = False
dropout = 0.1

# 2xA6000
# batch_size = 192
# gradient_accumulation_steps = 4

# 2xV100
batch_size = 48
gradient_accumulation_steps = 16
max_seq_len = 256
learning_rate = 8e-4

# At RTX4090
# batch_size = 24
# gradient_accumulation_steps = 16
