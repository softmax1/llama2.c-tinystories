wandb_project = "softermax-eval"
init_from = "resume"
eval_iters = 200

wandb_run_name = "softmax0-15m-2023_12_11_10_05_14-100k"
out_dir = f"out/{wandb_run_name}"
softmax1 = False
always_save_checkpoint = False
wandb_log = False

batch_size = 16
eval_only = True
device = "mps"
dtype = "float16"  # "float16"
