import transformers 
import datasets 

from transformers import (
    AutoTokenizer, AutoModel,  
    TFAutoModel, TFAutoModelForQuestionAnswering, 
)
from datasets import (
    concatenate_datasets, list_datasets, 
)
from pathlib import Path

def load_backbone(model_path, load_kwargs):
    is_local = Path(model_path).exists()
    backbone = TFAutoModel.from_pretrained(
        model_path, 
        local_files_only=is_local, 
        **load_kwargs, 
    )
    return backbone

def load_tokenizer(backbone_name, special_tokens): 
    return AutoTokenizer.from_pretrained(
        backbone_name, add_special_tokens=special_tokens
    )