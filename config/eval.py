wandb_project = "softermax-eval"
init_from = "resume"
eval_iters = 100

wandb_run_name = "softmax1-42m-2023_12_13_10_37_41-43k"
out_dir = f"out/{wandb_run_name}"
softmax1 = True
always_save_checkpoint = False
wandb_log = False

batch_size = 32
eval_only = True
device = "mps"
dtype = "float16"
