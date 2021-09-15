# LEARNINGS
- Adding even 0.05 dropout fucks up the performance totally. 
- WD: 1e-3 with LR: 1e-4 FAILED
- Try different negative weights, try 0.1
- Very low LR gives non zero accuracy as 0 and loss as NaN. Why?
- LR 5e-5 >> 1e-4 with WD: 1e-7. Very sensetive to LR
- In 4th epoch non_zero_accuracy always decreases. Fucking always
- Using lookahead in optimizers makes the results much worse

# TODO
- backbone dropout 0.01, no qa dropout
- hidden layer
- 1e-2, 1e-6 dropout with everything else same

# CURRENT RUNS 
__Fixed__

__Kaggle__


__Yash__
dropout: 0.01
WD: 3e-7
LR: 5e-5, 0.75

__Harshit__
dropout: 0.025
WD: 1e-7
LR: 3e-5, 1

# 13/09/21
Trying to establish a good RemBERT baseline 
trained on finetuning data with paper hyperparameters

__Rembert Baseline__
0.708, hi: 0.758, ta: 0.606
LR: 3e-5, WD: 1e-7
neg wt: 1, start wt: 1, concat: False, 
special_tokens: False
dropout: 0
data: base + 'mlqa_hi_translated', 'adversarial_qa', 'squad_v2'


__Rembert V2 Kaggle__
ja: xx, hi: xx, ta: xx
non_zero: 56, 36 | acc: 913, 913
LR: (1e-5, 1), WD: 1e-4, 
all external data availible
dropout: 0 everywhere
use special tokens, concat: True
neg wt: 0.75, start_wt: 1.5


__Rembert V3 Harshit__
0.693, hi: 759, ta: 0.558
non_zero: (46, 44) | (acc: 9182, 9180) 
all external data availible
dropout: 0 everywhere
use special tokens, concat: True
LR: (3e-5, 1), WD: 1e-6, 
neg wt: 0.5, start_wt: 1.5
- Try 1e-4 lr with 0.75 gamma
- split validation into paragrPh

__Rembert V3 Yash__
data: All external data
tokens+, concat+, neg_wt: 0.5, start wt: 1
LR: (1e-10, 1e-4, 0.75), Epochs: 4, CPE: 4
WD: 1e-7, dropout: 0 everywhere
seq: 512, 256, min_para_len: 256()

__Rembert V3 Kaggle__
- same as Yash
- LR: (1e-10, 5e-5, 1), Epochs: 4, CPE: 4


__Rembert V3 Harhit__
all external data, tokens+, concat+
dropout: 0, neg wt: 0.5, start_wt: 1
LR: (1e-10, 1e-8, 10), WD: 1e-2


__TODO__
best performing
negative weight: 0.1


Kaggle: WD: 1e-2, LR: 9e-6
Yash: WD: 1e-7, LR: 3e-5

# Roumd 1

# Round 2 
Squad Adversial + basic + mlqa hindi
# REMBERT 
- High dropout gave much worse results
- High weight decay gave bad results but it was due to high lr
- Comp gold performed better than comp clean / comp original

- Lookahead __ results
- Using fixed lr ___ 
- Low negative weight ___
- 3e-2 weight decay ___
- Adding tokens ___
- Adding confusing sentences ___
- Finetuning on hard negatives ___
- Adding external data for regularization ___