import tensorflow_addons as tfa
import tensorflow as tf

from termcolor import colored 
from pathlib import Path
import os

def lr_scheduler_factory(kwargs):
    if isinstance(kwargs, float): 
        print('Using constant learning rate')
        return kwargs
    elif kwargs.name == 'ExponentialCyclicalLearningRate': 
        print('Using exponential cyclic LR')
        return tfa.optimizers.ExponentialCyclicalLearningRate(
            initial_learning_rate=kwargs.min_lr,
            maximal_learning_rate=kwargs.max_lr, 
            gamma=kwargs.gamma, 
            step_size=kwargs.step_size, 
            scale_mode='cycle', 
        )


def optimizer_factory(kwargs, lr_scheduler): 
    optimizer_name = kwargs.name
    if optimizer_name == 'AdamW': 
        optimizer =  tfa.optimizers.AdamW(
            beta_1=kwargs.beta_1, 
            beta_2=kwargs.beta_2, 
            epsilon=kwargs.epsilon, 
            weight_decay=kwargs.wd, 
            clipnorm=kwargs.max_grad_norm, 
            amsgrad=False, # Does not work on TPU
            learning_rate=lr_scheduler, 
        )
    if kwargs.use_swa: 
        print(colored('Using SWA', 'red'))
        optimizer = tfa.optimizers.SWA(optimizer)
    if kwargs.use_lookahead: 
        print(colored('Using Lookahead', 'red'))
        optimizer = tfa.optimizers.Lookahead(optimizer)
    return optimizer

#############################
##### Callbacks Factory #####
#############################

def get_model_checkpoint_callback(checkpoint_dir, checkpoint_file, common_kwargs):
    # 'checkpoint-{epoch:02d}-{val_loss:.4f}.h5'
    print(f'Saving model checkpoints at {checkpoint_dir}/{checkpoint_file}')
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists(): 
        print(f'Created the folder: {checkpoint_dir}')
        os.makedirs(checkpoint_dir)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir/checkpoint_file),
        save_weights_only=True,
        save_best_only=True, 
        **common_kwargs, 
    )

def get_early_stopping_callback(patience, common_kwargs): 
    print(f'Will stop training if metric does not improve in {patience} epochs')
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        restore_best_weights=True, 
        **common_kwargs,
    )
    
def get_time_stopping_callback(max_train_hours):
    import tensorflow_addons as tfa 
    return tfa.callbacks.TimeStopping(
        seconds=max_train_hours*3600
    )

def get_wandb_callback(common_kwargs):
    # TODO: Add training data, or remove log grads
    from wandb.keras import WandbCallback
    return WandbCallback(
        monitor=common_kwargs['monitor'], 
        verbose=0, mode=common_kwargs['mode'], 
    #     save_weights_only=True, 
    #     log_gradients=False, 
    )


def get_reduce_lr_on_plateau(patience, factor, common_kwargs): 
    print(f'reducing lr by {factor} if metric does not improve within {patience} epochs')
    return tf.keras.callbacks.ReduceLROnPlateau(
        factor=factor,
        patience=patience,
        min_delta=0,
        min_lr=1e-8,
        **common_kwargs, 
    )
    
def get_tqdm_bar_callback(): 
    import tensorflow_addons as tfa
    return tfa.callbacks.TQDMProgressBar()

def get_terminate_on_nan_callback(): 
    return tf.keras.callbacks.TerminateOnNaN()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )

def callbacks_factory(kwargs):
    monitor, mode = kwargs.monitor_mode
    common_kwargs = {
        'monitor': monitor, 
        'mode': mode, 
        'verbose': True, 
    }
    callbacks = [
        get_model_checkpoint_callback(kwargs.checkpoint_dir, kwargs.checkpoint_file, common_kwargs)
    ]
    if 'early_stop' in kwargs: 
        callbacks.append(get_early_stopping_callback(kwargs.early_stop, common_kwargs))
    if 'max_train_hours' in kwargs: 
        callbacks.append(get_time_stopping_callback(kwargs.max_train_hours))
    if 'reduce_lr_patience' in kwargs: 
        callbacks.append(get_reduce_lr_on_plateau(kwargs.reduce_lr_patience, kwargs.reduce_lr_factor, common_kwargs)) 
    if 'wandb_callback' in callbacks and kwargs.wandb_callback: 
        try: 
            callbacks.append(get_wandb_callback(common_kwargs))
        except Exception as e:
            print(f'Skipping wandb callback coz:', e) 
    if 'use_tqdm_bar' in callbacks and kwargs.use_tqdm_bar: 
        callbacks.append(get_tqdm_bar_callback())
    if 'terminate_on_nan' in callbacks and kwargs.terminate_on_nan: 
        callbacks.append(get_terminate_on_nan_callback())
    return callbacks     

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')



# def tb(tb_dir, train_steps): 
#     start_profile_batch = train_steps+10
#     stop_profile_batch = start_profile_batch + 100
#     profile_range = f"{start_profile_batch},{stop_profile_batch}"
#     log_path = tb_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     tensorboard_callback = tf.keras.kwargs.TensorBoard(
#         log_dir=log_path, histogram_freq=1, update_freq=20,
#         profile_batch=profile_range, 
#     )
#     return tensorboard_callback

# def tensorboard_callback(log_dir):
#     os.makedirs(log_dir, exist_ok=True)
#     return tf.keras.callbacks.TensorBoard(
#         log_dir=str(log_dir)
#     )