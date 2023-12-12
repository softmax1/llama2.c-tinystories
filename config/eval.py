from datetime import datetime

wandb_project = "softermax-eval"
init_from = "resume"

# For softmax0
wandb_run_name = "softmax1-15m-2023_08_25_11_47_04"
out_dir = f"out/{wandb_run_name}"
softmax1 = False
always_save_checkpoint = False
wandb_log = False

batch_size = 32
eval_only = True
device = "cpu"
dtype = "qint8"  # "float16"
