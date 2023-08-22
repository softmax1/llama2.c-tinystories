# some janitorial work for WandB + HuggingFace
import wandb, os, huggingface_hub

def login_all(do_wandb=True, do_hf=True):
	if do_wandb and 'WANDB_API_KEY' in os.environ and os.environ['WANDB_API_KEY']:
		print("Logging in to WandB... 📈 ")
		wandb.login(key=os.environ['WANDB_API_KEY'])
	else:
		print("No WandB API key found. Skipping login...")
		
	if do_hf and 'HF_TOKEN' in os.environ and os.environ['HF_TOKEN']:
		print("Logging in to HuggingFace Hub... 🤗 ")
		huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)
	else:
		print("No HuggingFace token found. Skipping login...")
