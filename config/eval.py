wandb_project = "softermax-eval"
init_from = "resume"
eval_iters = 20

wandb_run_name = "softmax1-42m-2023_12_13_10_37_41-43k/iter_70000"
out_dir = f"out/{wandb_run_name}"
softmax1 = True
always_save_checkpoint = False
wandb_log = False

batch_size = 32
eval_only = True
device = "cpu"
dtype = "qint8"
# device = "mps"
# dtype = "float16"
