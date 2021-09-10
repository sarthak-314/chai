# ABOUT
- finetuning for SQuAD: LR: 9e-6, batch size: 128, epochs: 3
- does not use dropout to increase model capacity. Not using dropout gives better results
- "dropout can hurt performance in large Transformer-based models"
- Rebalanced mBERT
- Max sequence length is 512?
- Apparantly input and output embeddings share weights ?
- Using larger output embeddings is good (768 in bert) is good because it prevents over specializing to pre training task
- Paper HP: 2e-4 lr with 2048 batch, 0.01 WD, 1 grad norm,  


# TRAINING HYPERPARAMETERS 

## Baseline Model (from paper)
__Pretraining__
- dropout: 0, wd: 1e-2, warmup: 15k steps, batch: 2048
__Squad Finetuning__
- batch_size: 128, lr: 9e-6, epochs: 3, max_grad: 1

## Round 1: Pretraining 
jaccard: 0.7xx, hindi: 0.8xx, tamil: 0.7xx

dropout: 0
external data: all, comp data: gold + tokens
wd: 1e-2, lr: (0, 2e-5, 1), epochs: 3 ** 4
negative: 0.75, start: 1.5

- High dropout gave much worse results
- High weight decay gave bad results but it was due to high lr
- Comp gold performed better than comp clean / comp original

## Round 2: Finetuning
jaccard: 0.7xx, hindi: 0.8xx, tamil: 0.7xx

comp data: xxx, external: hi-en-ta data
wd: 3e-2, lr: (0, 1e-5, 1), epochs: 3 ** 4
negative: 0.2, start: 1.5, max_grad: 10

- Lookahead __ results
- Using fixed lr ___ 
- Low negative weight ___
- 3e-2 weight decay ___
- Adding tokens ___
- Adding confusing sentences ___
- Finetuning on hard negatives ___
- Adding external data for regularization ___
"""

MODEL = {
    'name': 'google/rembert', 
    'dropout': 0, 
    'max_seq_len': 512, 'doc_stride': 192, 
}

DATA = {
    'external': [
        'squad_v2', 'mlqa_hi_full', 'tydiqa_goldp', 
        'mlqa_hi_translated', 
        'squad_ta_3k', 'mlqa_hi_en', 'xquad', 
    ], 
}

COMP_DATA = {
    'add_word_token': True, 
    'remove_multi_answer_sentences': True, 
    'use_goldp': True, # Take first gold passage for answer

    # Competition Data Augmentations
    'use_hard_negatives': False, # Replace uniform with hard negatives
    'add_confusing_sentences': False, # https://arxiv.org/pdf/1707.07328.pdf
    # TODO: Use addsent to add confusing sentences
}

COMPILE = {
    'negative_weight': 1, 
    'start_weight': 1.5,
    'optimizer': {
        'name': 'AdamW', 
        'weight_decay': 1e-2, 'max_grad_norm': 1,
        'use_swa': True, 
        'use_lookahead': False, 
    }, 
    'steps_per_execution': 128, 
}

TRAIN = {
    'max_epochs': 16, 
    'checkpoints_per_epochs': 4, 
    'early_stopping': 2, 
    'lr': { 'init_lr': 1e-6, 'max_lr': 3e-5, 'gamma': 1 }, 
}

"""
CURRENT RUNS
------------
Fixed: google/rembert, full data training
Kaggle:  
Yash:  
Harshit:  
Colab:  

 IDEAS
-------
- Jaccard based loss function?
    https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477
- Smoothed CE 


=========
| TODOS |
=========

GENERAL
-------
- Go through all the solutions of commonlit, tweet sentiment and other nlp comps
- Expand the dataset on lowest scoring pairs
- Post processing 
- Pre processing
- Additional Datasets ? 
- Change max_seq after finetuning? 
- Beam Search ? 
- Write code in lightning
- longformer-4096 for tf
- concat start and end 738 to 1536
- Remove grad clip when finetuning? 
- add checkpoints_per_epoch 

DATASETS
--------
(For final tuning have hard positive and hard negatives)
- comp_hard_negatives
    Find all the hard negatives
- comp_adversial_sentence
    - Adversially inserted sentences for correct predictions and expansion for incorrect predictions
        -- https://arxiv.org/pdf/1707.07328.pdf
        -- Example: add all the sentences about years into the golds sentence
    - Sentence Order shuffling as augmentation

INFERENCE
---------
- Change to 16 bit precision
- Change doc stride to 256 to reduce inference time in half
- max_contents to take first 10k or 20 - 30k words in a article

FUTURE
------
- Knowledge distilation with mt5-xxl
- ML tools, hparams logging
- go through transformers code, TFTrainer
- ensemble with diverse strides and shit
"""

"""
RemBERT
-------

"""