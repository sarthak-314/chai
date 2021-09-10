import transformers 
import datasets

from transformers import (
    AutoTokenizer, TFAutoModel, TFAutoModel, TFAutoModelForQuestionAnswering, 
    EvalPrediction, 
)
from datasets import (
    concatenate_datasets, list_datasets, 
)
def _get_word_len_tokens(word_lens): 
    return [f'[WORD={word_len}]' for word_len in word_lens]
def _get_context_len_tokens(context_lens): 
    return [f'[CONTEXT={context_len}]' for context_len in context_lens]

WORD_LENS =[0, 10, 20, 50, 100, 200, 400, 1000, 2000, 4000, 10000, 25000]
CONTEXT_LENS = [500, 1000, 2000, 4000, 8000, 16000]

WORD_LEN_TOKENS = _get_word_len_tokens(WORD_LENS)
CONTEXT_LEN_TOKENS = _get_context_len_tokens(CONTEXT_LENS)



