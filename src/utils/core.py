"""
Core constants, classes and functions that can be used in other modules
- HP: dict wrapper for HyperParameters 
- colors: color code print messages
"""


from collections import defaultdict 
from termcolor import colored
from pathlib import Path
import pickle
import torch 
import os 

from IPython.display import display, Markdown

# Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')

class HP(defaultdict): 
    """
    dict wrapper for HyperParameters 
    TODO: Don't be a perfectionist. Effort put in this was not worth it
    """
    def __init__(self, base_dict={}): 
        super().__init__()
        self._add_base_dict_to_self(base_dict)
        
    def _add_base_dict_to_self(self, base_dict): 
        for key, val in base_dict.items(): 
            self[key] = val

    def __getattr__(self, key):
        if key in self: 
            return self[key]
        else: 
            raise Exception(f'{key} not found in the object')

    def __setattr__(self, key, value):
        self[key] = value
    
    def __repr__(self):
        dict_key_color = 'red'
        dict_val_color = 'blue'
        res = '\n'
        for key, value in self.items(): 
            k, v = str(key)[:512], str(value)[:512]
            if isinstance(value, HP): 
                space = min(8, len(k)+1)
                v = v.replace('\n', '\n' + ' '*space)
                k = '\n' + colored(k, dict_key_color, attrs=['bold'])
            else: 
                k = colored(k, dict_key_color)
            line = k + ': ' + colored(v, dict_val_color)
            res += line + '\n'
        return res

    def save(self, path): 
        with open(path, 'wb') as f:
            pickle.dump(self, f)        

    @staticmethod
    def load(path): 
        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file

    def _heading(self, title, level=3): 
        margin = 'margin-left:5px;'
        try: 
            text = title.title()
            html = f"<h{level} style='text-align:center; {margin}'> {text} </h{level}> <hr/>" 
            print()
            display(Markdown(html))
        except Exception as e: 
            print(e)

    def display(self, title): 
        self._heading(title)
        print(self)



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
print('Notebook running on', colored(ENV, 'blue'), 'on', colored(HARDWARE, 'blue'))


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