import os
try: 
    import wandb
except: 
    print('WandB not found. Attempting to install wandb')
    os.system('pip install wandb')
    
# Constants for this Project


    
def _login_to_wandb(): 
    api_key = os.get_env('WANDB_API_KEY', None)
    if api_key is None: 
        print('Set WANDB_API_KEY environment variable')
    wandb.login()