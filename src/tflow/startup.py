import tensorflow_addons as tfa
import tensorflow as tf

from termcolor import colored 
from pathlib import Path

from src.utils.core import WORKING_DIR

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }
TB_DIR = WORKING_DIR / 'tb-logs'

# Startup Functions
def _enable_mixed_precision(): 
    """
    Can sometimes lead to poor / unstable convergence
    """
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def set_jit_compile(enable_jit=True):
    """
    - When using dynamic sizes, compile time can add up
    - Uses extra memory
    - Don't use for short scripts
    https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
    """
    if enable_jit:  
        print('Using JIT compilation')
        tf.config.optimizer.set_jit(True)
    else: 
        tf.config.optimizer.set_jit(False)
    

def tf_accelerator(bfloat16, jit_compile):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    if bfloat16: 
        _enable_mixed_precision()
        print('Mixed precision enabled')
    set_jit_compile(jit_compile)
    return strategy


# Common Notebook Functions
def save_weights(model, filepath):
    filepath = str(filepath)
    print('Saving model weights at', colored(filepath, 'blue'))
    model.save_weights(filepath)

def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)