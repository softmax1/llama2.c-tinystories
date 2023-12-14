from datetime import datetime

wandb_project = "softmax1-tinystories"
checkpoint_intervals = (1000, 2000, 4000, 8000, 16000, 32000)

# For softmax0
wandb_run_name = "softmax0-42m-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"out/{wandb_run_name}"
softmax1 = False

# 42M model
dim = 512
n_layers = 8
n_heads = 8
n_kv_heads = 8
compile = False
dropout = 0.1

# 2xV100
device = "cuda"
batch_size = 56
gradient_accumulation_steps = 12
max_seq_len = 256
learning_rate = 8e-4
