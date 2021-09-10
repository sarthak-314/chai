from termcolor import colored 
from pathlib import Path
import torch 
import os 

# Solve Environment, Hardware & Online Status
def _solve_env(): 
    if 'KAGGLE_CONTAINER_NAME' in os.environ: 
        return 'Kaggle'
    elif Path('/content/').exists(): 
        return 'Colab'
    else: 
        return 'Local'
def _solve_hardware(): 
    if torch.cuda.is_available(): 
        print('GPU Device:', colored(torch.cuda.get_device_name(0), 'green'))
        return 'GPU'
    elif 'TPU_NAME' in os.environ: 
        return 'TPU'
    else: 
        return 'CPU'
def _solve_online_status(): 
    try: 
        os.system('pip install wandb')
        return True
    except: 
        return False
    
ENV = _solve_env()
HARDWARE = _solve_hardware()
IS_ONLINE = _solve_online_status()
print('Notebook running on', colored(ENV, 'blue'), 'on ', colored(HARDWARE, 'blue'))


# Useful Paths for each environment
KAGGLE_INPUT_DIR = Path('/kaggle/input')
if ENV == 'Colab': 
    WORKING_DIR = Path('/content')
    TMP_DIR = Path('/content/tmp')
elif ENV == 'Kaggle': 
    WORKING_DIR = Path('/kaggle/working')
    TMP_DIR = Path('/kaggle/tmp')
else: 
    WORKING_DIR = Path('C:/Users/sarth/Desktop/chai')
    TMP_DIR = WORKING_DIR / 'tmp'

# Check if repo is loaded correctly
def _ensure_repo_dir_is_correct(): 
    if ENV == 'Kaggle': 
        assert Path('/kaggle/working/temp').exists(), red('Wrong Repo Directory')
    elif ENV == 'Colab': 
        assert Path('/content/temp').exists(), red('Wrong Repo Directory')
_ensure_repo_dir_is_correct()
    
# Mount Drive in Colab
def _mount_drive(): 
    from google.colab import drive
    drive.mount('/content/drive')
if ENV == 'Colab': 
    _mount_drive()


# Termcolor Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')
