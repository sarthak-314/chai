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

import tensorflow as tf
import torch

sys.path += [
    '/kaggle/working/chai', 
    '/content/chai', 
]
import src 
import src.utils
from src.utils.core import *


# Uncommonly Used Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import warnings
import shutil
import math
import glob
import sys

# Jupyter Notebook Setup
def _setup_jupyter_notebook(): 
    from IPython.core.interactiveshell import InteractiveShell
    from IPython import get_ipython
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
    os.chdir(WORKING_DIR/'temp')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'temp'))
    os.chdir(WORKING_DIR)

def clone_repo(repo_url): 
    # TODO: Should this function exist?
    repo_name = repo_url.split('/')[-1].replace('-', '_')
    clone_dir = str(WORKING_DIR/repo_name)
    subprocess.run(['git', 'clone', repo_url, clone_dir])
    os.chdir(clone_dir)
    sys.path.append(clone_dir)
    print(f'Repo {repo_url} cloned')    


# Competition Specific Constants & Functions
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')
DF_DIR = {
    'Kaggle': KAGGLE_INPUT_DIR/'chai-dataframes', 
    'Colab': DRIVE_DIR/'Dataframes'
}[ENV]