# some janitorial work for WandB + HuggingFace
import wandb, os, huggingface_hub
from dotenv import load_dotenv
load_dotenv()
wandb.login(key=os.environ['WANDB_API_KEY'])
huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)

softmax1 = True
device = 'cuda'
compile = False
wandb_project = 'softmax1-tinystories'
wandb_log = True
n_heads = 12
n_layers = 12
dim = 768
max_seq_len = 1024