# some janitorial work for WandB + HuggingFace
import wandb, os, huggingface_hub

def login_all(do_wandb=True, do_hf=True):
	if do_wandb and 'WANDB_API_KEY' in os.environ and os.environ['WANDB_API_KEY']:
		print("Logging in to WandB... 📈 ")
		try:
			wandb.login(key=os.environ['WANDB_API_KEY'])
		except Exception as e:
			print(f"Invalid WandB key: {e}")
	else:
		print("No WandB API key found. Skipping login...")
		
	if do_hf and 'HF_TOKEN' in os.environ and os.environ['HF_TOKEN']:
		print("Logging in to HuggingFace Hub... 🤗 ")
		try:
			huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)
		except Exception as e:
			print(f"Invalid HuggingFace token: {e}")
	else:
		print("No HuggingFace token found. Skipping login...")

if __name__ == '__main__':
	login_all()