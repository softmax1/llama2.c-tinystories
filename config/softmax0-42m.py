from datetime import datetime

wandb_project = "softmax1-tinystories"

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

# TPU v2-8
batch_size = 192
gradient_accumulation_steps = 4
max_seq_len = 256
learning_rate = 8e-4

device = "xla"  # tpu
dtype = "bfloat16"
