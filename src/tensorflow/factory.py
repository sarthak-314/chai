import tensorflow_addons as tfa
import tensorflow as tf
from termcolor import colored 

def lr_scheduler_factory(lr_scheduler_kwargs): 
    if isinstance(lr_scheduler_kwargs, float): 
        print(colored('Using constant learning rate', 'yellow'))
        return lr_scheduler_kwargs
    lr_scheduler = tfa.optimizers.ExponentialCyclicalLearningRate(
        initial_learning_rate=lr_scheduler_kwargs['init_lr'], 
        maximal_learning_rate=lr_scheduler_kwargs['max_lr'], 
        step_size=lr_scheduler_kwargs['step_size'], 
        gamma=lr_scheduler_kwargs['gamma'], 
    )
    return lr_scheduler

def optimizer_factory(optimizer_kwargs, lr_scheduler): 
    optimizer_name = optimizer_kwargs['name']
    if optimizer_name == 'AdamW': 
        optimizer = tfa.optimizers.AdamW(
            weight_decay=optimizer_kwargs['weight_decay'],
            learning_rate=lr_scheduler,  
            amsgrad=False, 
            clipnorm=optimizer_kwargs['max_grad_norm'], 
            epsilon=optimizer_kwargs['epsilon'], 
            beta_2=optimizer_kwargs['beta2'], 
        )
    elif optimizer_name == 'Adagrad': 
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=lr_scheduler, 
        )
        print('Skipping weight decay for Adagrad')
    if optimizer_kwargs['use_lookahead']: 
        print(colored('Using Lookahead', 'red'))
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if optimizer_kwargs['use_swa']: 
        print(colored('Using SWA', 'red'))
        optimizer = tfa.optimizers.SWA(optimizer)
    return optimizer