"""
Startup script to run in the beggining of every Jupyter Notebook for the competition
- Import common libraries
- Jupyter Notebook Setup: autoreload, display all, add to sys.path
- Import common functions, classes & constants
- Import competition specific functions / constants
"""
from __future__ import absolute_import

# Commonly Used Libraries
from IPython.display import clear_output 
from functools import partial
from termcolor import colored
from tqdm.auto import tqdm
from pathlib import Path 
from time import time
import pandas as pd
import numpy as np
import pickle
import random
import json
import cv2
import sys
import gc
import os

from omegaconf import OmegaConf
import omegaconf

import tensorflow as tf
import torch

sys.path += [
    '/kaggle/working/chai', 
    '/content/chai', 
    'C:\\Users\\sarth\\Desktop\\chaiv3', 
]
import src 
import src.utils
from src.utils.core import *


# Uncommonly Used Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython import get_ipython
from PIL import Image
import subprocess
import warnings
import shutil
import math
import glob

# Jupyter Notebook Setup
def _setup_jupyter_notebook(): 
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
    ipython = get_ipython()
    try: 
        ipython.magic('matplotlib inline')
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
    except: 
        print('could not load ipython magic extensions')
_setup_jupyter_notebook()

def _ignore_deprecation_warnings(): 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
_ignore_deprecation_warnings()

# Startup Notebook Functions
REPO_PATH = 'https://github.com/sarthak-314/chai'
def sync(): 
    # TODO: Do everything locally once you get a GPU
    'Sync Notebook with VS Code'
    os.chdir(WORKING_DIR/'chai')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'chai'))
    os.chdir(WORKING_DIR)

def clone_repo(repo_url): 
    # TODO: Should this function exist?
    repo_name = repo_url.split('/')[-1].replace('-', '_')
    clone_dir = str(WORKING_DIR/repo_name)
    subprocess.run(['git', 'clone', repo_url, clone_dir])
    os.chdir(clone_dir)
    sys.path.append(clone_dir)
    print(f'Repo {repo_url} cloned')    

# Check if repo is loaded correctly
def _ensure_repo_dir_is_correct(): 
    if ENV == 'Kaggle': 
        assert Path('/kaggle/working/chai').exists(), red('Wrong Repo Directory')
    elif ENV == 'Colab': 
        assert Path('/content/chai').exists(), red('Wrong Repo Directory')
_ensure_repo_dir_is_correct()
    
# Mount Drive in Colab
def _mount_drive(): 
    from google.colab import drive
    drive.mount('/content/drive')
    
if ENV == 'Colab': 
    _mount_drive()


# Competition Specific Constants & Functions
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')
DF_DIR = {
    'Kaggle': KAGGLE_INPUT_DIR/'chai-dataframes', 
    'Colab': DRIVE_DIR/'Dataframes', 
    'Local': Path('C:\\Users\\sarth\\Desktop\\chaiv3\\data')
}[ENV]

def get_word_len_tokens(word_lens): 
    return [f'[WORD={word_len}]' for word_len in word_lens]
def get_context_len_tokens(context_lens): 
    return [f'[CONTEXT={context_len}]' for context_len in context_lens]

# HYPERPARAMETERS 
def heading(title, level=3):
    margin = 'margin-left:5px;'
    try: 
        text = title.title()
        html = f"<h{level} style='text-align:center; {margin}'> {text} </h{level}> <hr/>" 
        print()
        display(Markdown(html))
    except Exception as e: 
        print(e)

def display_hparams(hp): 
    heading('Hyperparameters')
    for key, value in hp.items(): 
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            print()
            print(colored(key, 'red', attrs=['bold']))
            space = min(4, len(key)+1)
            for k, v in value.items(): 
                print(' '*space, colored(k+':', 'red'), colored(v, 'blue'))
            print()
        else:
            print(colored(key+':', 'red'), colored(value, 'blue'))
    
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def hyperparameters(line, cell):
    'Magic command to write hyperparameters into a yaml file and load it with hydra'
    print(f'Writing the hyperparameters to {line}')
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
    hyperparameters_config = OmegaConf.load(line)
    get_ipython().user_ns['HP'] = hyperparameters_config
    print('Hyperparameters loaded in the variable HP')
    display_hparams(hyperparameters_config)