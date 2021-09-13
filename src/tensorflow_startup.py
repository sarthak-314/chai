# Try to install tensorflow_addons if not already installed in the notebook. TODO: Is this the best way to do this?
try: 
    import tensorflow_addons as tfa 
except: 
    print('tensorflow_addons not found. Trying to install tensorflow_addons')
    import os
    os.system('pip install tensorflow_addons')

import tensorflow as tf
from src.utils.core import HARDWARE, WORKING_DIR
from src.tensorflow_factory import lr_scheduler_factory, optimizer_factory, callbacks_factory

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }
TB_DIR = WORKING_DIR / 'tb-logs'

# Startup Functions
def _enable_mixed_precision(): 
    """
    - TODO: Disable when finetuning because "it can sometimes lead to poor / unstable convergence" ??
    """
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def set_jit_compile(enable_jit=True):
    """
    https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
    - Don't use variable sizes with TPU. Compile time adds up
    - Uses extra memory
    - Don't use for short scripts
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
    
    if HARDWARE is 'CPU': 
        print('CPU detected. Skipping mixed precision and jit compilation')
        return strategy
    
    if bfloat16: 
        _enable_mixed_precision()
        print('Mixed precision enabled')
    set_jit_compile(jit_compile)
    return strategy


# Common Notebook Functions
def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)
