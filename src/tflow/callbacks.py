from pathlib import Path
import tensorflow as tf 
import datetime
import os


def get_model_checkpoint_callback(checkpoint_dir, checkpoint_file, common_kwargs):
    print(f'Saving model checkpoints at {checkpoint_dir}/{checkpoint_file}')
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists(): 
        print(f'Created the folder: {checkpoint_dir}')
        os.makedirs(checkpoint_dir)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir/checkpoint_file,
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
    from wandb.keras import WandbCallback
    return WandbCallback(
        monitor=common_kwargs['monitor'], 
        verbose=0, mode=common_kwargs['mode'], 
        save_weights_only=True, 
        log_gradients=True, 
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
    
def tqdm_bar(): 
    import tensorflow_addons as tfa
    return tfa.callbacks.TQDMProgressBar()

def terminate_on_nan(): 
    return tf.keras.callbacks.TerminateOnNaN()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )

def callbacks_factory(callbacks): 
    callbacks_list = [
        get_model_checkpoint_callback(
            callbacks.checkpoint_dir, callbacks.checkpoint_file, callbacks.common_kwargs, 
        ), 
        get_early_stopping_callback(callbacks.early_stop, callbacks.common_kwargs), 
        get_time_stopping_callback(callbacks.max_train_hours), 
        get_reduce_lr_on_plateau(
            callbacks.reduce_lr_patience, callbacks.reduce_lr_factor, callbacks.common_kwargs
        ), 
    ]
    if callbacks.wandb_callback: 
        wandb_callback = get_wandb_callback(callbacks.common_kwargs)
        callbacks_list.append(wandb_callback)
    return callbacks_list     

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

# def tb(tb_dir, train_steps): 
#     start_profile_batch = train_steps+10
#     stop_profile_batch = start_profile_batch + 100
#     profile_range = f"{start_profile_batch},{stop_profile_batch}"
#     log_path = tb_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=log_path, histogram_freq=1, update_freq=20,
#         profile_batch=profile_range, 
#     )
#     return tensorboard_callback

# def checkpoint(checkpoint_dir=None):
#     # checkpoint_filepath = 'checkpoint-{epoch:02d}-{val_loss:.4f}.h5'
#     checkpoint_filepath = 'checkpoint.h5'
#     if checkpoint_dir is not None: 
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         checkpoint_filepath = checkpoint_dir / checkpoint_filepath
#     return tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_filepath,
#         save_weights_only=True,
#         save_best_only=True, 
#         **common_kwargs, 
#     )

# def early_stop(patience=3):
#     return tf.keras.callbacks.EarlyStopping(
#         patience=patience, 
#         restore_best_weights=True, 
#         **common_kwargs,
#     )



# def tensorboard_callback(log_dir):
#     os.makedirs(log_dir, exist_ok=True)
#     return tf.keras.callbacks.TensorBoard(
#         log_dir=str(log_dir)
#     )



