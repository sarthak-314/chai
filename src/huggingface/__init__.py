import transformers 
import datasets

from transformers import (
    AutoTokenizer, TFAutoModel, TFAutoModel, TFAutoModelForQuestionAnswering, 
    EvalPrediction, 
)
from datasets import (
    concatenate_datasets, list_datasets, 
)
def get_word_len_tokens(word_lens): 
    return [f'[WORD={word_len}]' for word_len in word_lens]
def get_context_len_tokens(context_lens): 
    return [f'[CONTEXT={context_len}]' for context_len in context_lens]