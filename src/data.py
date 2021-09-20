from termcolor import colored
from functools import partial
import tensorflow as tf
from time import time 
import pandas as pd
import numpy as np
import random
import os

import datasets
from datasets import concatenate_datasets
from src.utils.core import red, blue, green

class CompetitionDataModule: 
    """
    Wrapper class for competition related QA dataframes
    - Read, clean & process competition dataframes
    - Add special tokens, transform into goldP + uniform negative 
    """
    def __init__(
        self, df_dir, fold, versions=[],  
        word_counts=None, context_lengths=None, 
        word_count_tokens_position=None, context_length_tokens_position=None, 
    ): 
        self.df_dir = df_dir 
        self.fold = fold
        self.versions = versions
        
        self.word_counts = word_counts 
        self.context_lengths = context_lengths 
        
        self.word_count_tokens_position = word_count_tokens_position
        self.context_length_tokens_position = context_length_tokens_position
    
    
    # --- PUBLIC API ---
    def prepare_original_dataframe(
        self, goldp_min_para_len=128, 
        original_skip_duplicate_ans_para=False, 
    ):
        start_time = time()
        df = pd.read_csv(self.df_dir/'comp_org.csv')
        df = self._clean_kaggle_noise(df)
        df = self.add_special_tokens(df)
        train, valid = df[df.fold!=self.fold], df[df.fold==self.fold]
        if 'goldp' in self.versions: 
            print(blue('GoldP found in competition dataframe versions. Converting to goldp'))
            gold = self.convert_to_goldp(train)
            negative = self.convert_to_uniform_negative(train, goldp_min_para_len)
            train = pd.concat([gold, negative])
        if 'original' in self.versions: 
            print(blue('Original dataframe found in competition dataframe versions'))
            org_train = df[df.fold!=self.fold]
            train = pd.concat([train, org_train])
        print(colored(time()-start_time, 'blue'), 'seconds to prepare original dataframe')
        print('train, valid: ', len(train), len(valid))
        return train, valid
    
    
    # --------------- ADD SPECIAL TOKENS ---------------
    def add_special_tokens(self, df): 
        if self.word_count_tokens_position is not None:
            print(f'Adding word count tokens to the dataframe at the beggining of every {self.word_count_tokens_position}')
            df = self.add_special_word_count_tokens(df, self.word_count_tokens_position)
        if self.context_length_tokens_position is not None: 
            print(f'Adding context length tokens to dataframe at the beggining of every {self.context_length_tokens_position}')
            df = self.add_special_context_len_tokens(df, self.context_length_tokens_position)
        return df
    
    def add_special_word_count_tokens(self, df, word_count_tokens_position): 
        new_contexts = []
        for i, row in df.iterrows(): 
            split_context_on = '\n'
            if word_count_tokens_position == 'sentence': 
                split_context_on = '।' if row.language == 'hindi' else '.'
            
            word_count_counter, split_contents = 0, []
            for content in row.context.split(split_context_on): 
                content = self._add_word_count_token(content, word_count_counter)
                word_count_counter += len(content)
                split_contents.append(content)
            
            new_context = split_context_on.join(split_contents)
            new_contexts.append(new_context)
        
        df['context'] = new_contexts
        df = self.fix_start(df)
        return df
        
    def _add_word_count_token(self, content, word_count):
        if self.word_count_tokens_position is not None: 
            word_token_num = self._closest_lower_number(self.word_counts, word_count)
            word_token = f'[WORD={word_token_num}]'
            content = word_token + content
        return content
    
    def add_special_context_len_tokens(self, df, context_length_tokens_position): 
        new_contexts = []
        for i, row in df.iterrows(): 
            split_context_on = '\n'
            if context_length_tokens_position == 'sentence': 
                split_context_on = '।' if row.language == 'hindi' else '.'
            split_contents = []
            for content in row.context.split(split_context_on): 
                content = self._add_context_length_token(content, len(row.context))
                split_contents.append(content)
            new_context = split_context_on.join(split_contents)
            new_contexts.append(new_context)
        df['context'] = new_contexts
        df = self.fix_start(df)
        return df
    
    def _add_context_length_token(self, content, context_length): 
        if self.context_length_tokens_position is not None: 
            context_token_num = self._closest_lower_number(self.context_lengths, context_length)
            context_token = f'[CONTEXT={context_token_num}]'
            content = context_token + content                    
        return content
    
    def _closest_lower_number(self, nums, target): 
        'Find the closest number in nums that is lower than target'
        for lower, upper in zip(nums, nums[1:]): 
            if upper > target: 
                return lower
        return nums[-1]
    
    
    # --------------- DATAFRAME TRANSFORMATIONS ---------------
    def convert_to_goldp(self, df): 
        'Only select golden paragraphs from the dataframe'
        gold = {'goldp': [], 'id': []}
        for i, row in df.iterrows(): 
            word_count = 0
            for paragraph in row.context.split('\n'): 
                if row.answer_text not in paragraph:
                    word_count += len(paragraph)
                    continue
                gold['goldp'].append(paragraph)
                gold['id'].append(row.id)
                break
        gold = pd.DataFrame.from_dict(gold)
        gold = df.merge(gold)
        gold['context'] = gold['goldp']
        del gold['goldp']
        gold = self.fix_start(gold)
        print(f"Taking {colored(len(gold), 'green')} golden paragraphs from the dataframe")
        return gold
        
    def convert_to_uniform_negative(self, df, min_paragraph_len=64): 
        'Only take the paragraphs that do not contain the answer text'
        negative = {'negative_para': [], 'id': []}
        for i, row in df.iterrows(): 
            word_count = 0
            for paragraph in row.context.split('\n'): 
                if row.answer_text in paragraph: continue
                if len(paragraph) < min_paragraph_len: continue
                negative['negative_para'].append(paragraph)
                negative['id'].append(row.id)
                word_count += len(paragraph)
        
        negative = pd.DataFrame(negative)
        negative = df.merge(negative)
        negative['context'] = negative['negative_para']
        del negative['negative_para']
        negative['answer_text'] = ''
        negative['answer_start'] = 0
        
        print(f"Taking {colored(len(negative), 'green')} negative paragraphs from the dataframe")
        return negative
    
    
    # --------------- HELPER FUNCTIONS ---------------
    def _clean_kaggle_noise(self, df): 
        'Semi-cleaned version of competition dataframe'
        df = df.set_index('id')
        df.loc['1a2160a69', 'answer_text'] = 'किर्काल्दी'
        df.loc['632c16ba0', 'answer_text'] = 'चार जोड़े'
        df = df[df.index!='2b41f3744']
        df = df[df.index!='bc9f0d533'] 
        df = df.reset_index()
        df = self.fix_start(df)
        return df
    
    def fix_start(self, df): 
        def func(row): 
            return row.context.find(row.answer_text) 
        df['answer_start'] = df.apply(func, axis=1)
        return df



class HuggingfaceDataModule: 
    """
    Wrapper around competition dataframes and, external huggingface datasets
    - Download, process & tokenize from huggingface datasets
    - Clean and save tokenized datasets to disk
    """
    def __init__(
        self, df_dir, fold, 
        tokenizer, max_seq_len, doc_stride, 
        verbose=True, force=True, 
    ): 
        self.df_dir = df_dir
        self.fold = fold
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        
        self.verbose = verbose
        self.force = force
        if not self.force: 
            print(red('Warning: Loading from cache preprocessed datasets'))
        self.num_workers = 4
        
        self.raw_datasets = {}
        self.processed_datasets = {}
        
        self.mlqa_hi_full_configs = [
            'mlqa.ar.hi', 'mlqa.vi.hi',  'mlqa.zh.hi', 'mlqa.es.hi', 
            'mlqa.hi.ar', 'mlqa.hi.de', 'mlqa.hi.vi', 'mlqa.hi.zh', 'mlqa.hi.es', 
            'mlqa.en.ar', 'mlqa.en.en',
            'mlqa.hi.hi', 'mlqa.hi.en', 'mlqa.en.hi', 
        ]
        
        
    def build_datasets(self, train, valid, external_datasets, cache_path=None): 
        start_time = time()
        self.raw_datasets = {
            'train': self.df_to_dataset(train), 
            'valid': self.df_to_dataset(valid), 
            'test': self.df_to_dataset(valid), 
        }
        self.print_dataset_info(self.raw_datasets['train'], 'Competition Train')
        
        # Load processed Dataset 
        train_datasets = []
        for dataset_name in external_datasets: 
            if (cache_path/dataset_name).exists() and not self.force: 
                print(f'Loading {dataset_name} from cache')
                processed_dataset = datasets.Dataset.load_from_disk(str(cache_path/dataset_name))
            else: 
                # Build Raw Dataset
                raw_dataset = self.load_external_dataset(dataset_name)
                self.raw_datasets[dataset_name] = raw_dataset.shuffle()
                if self.verbose: 
                    self.print_dataset_info(raw_dataset, dataset_name)
                
                # Build & Save processed dataset
                print('Building processed datasets and saving it in current folder')
                processed_dataset = self.process_and_save_dataset(
                    self.raw_datasets[dataset_name], dataset_name, is_train=True, 
                )
                
            train_datasets.append(processed_dataset.shuffle())
        train_datasets.append(self.process_and_save_dataset(self.raw_datasets['train'], 'train', is_train=True))
        
        self.processed_datasets = {
            'train': concatenate_datasets(train_datasets).shuffle(), 
            'valid': self.process_and_save_dataset(self.raw_datasets['valid'], 'valid', is_train=True), 
            'test': self.process_and_save_dataset(self.raw_datasets['valid'], 'test', is_train=False), 
        }
        print(blue(time()-start_time), 'seconds to build processed datasets')
        return self.processed_datasets
        
    def process_and_save_dataset(self, raw_dataset, dataset_name, is_train=True): 
        prepare_features_fn = HuggingfaceDataModule.prepare_train_features if is_train \
        else HuggingfaceDataModule.prepare_valid_features
        prepare_features = partial(
            prepare_features_fn, 
            tokenizer=self.tokenizer, 
            max_seq_len=self.max_seq_len, 
            doc_stride=self.doc_stride, 
        )
        start_time = time()
        processed_dataset = raw_dataset.map(
            prepare_features, 
            batched=True, 
            num_proc=self.num_workers, 
            remove_columns=raw_dataset.column_names, 
        )
        print(colored(time()-start_time, 'blue'), 'seconds to process the dataset')
        os.makedirs('datasets', exist_ok=True)
        processed_dataset.save_to_disk(f'datasets/{dataset_name}')
        print(f'Processed dataset saved to {dataset_name}')
        return processed_dataset
        
    
    def load_external_dataset(self, dataset_name): 
        if dataset_name  == 'tydiqa_goldp': 
            return self.load_splits('tydiqa', 'secondary_task')
        elif dataset_name == 'squad_v2': 
            return self.load_splits('squad_v2')
        elif dataset_name == 'adversarial_qa': 
            return self.load_splits('adversarial_qa', 'adversarialQA')
        elif dataset_name == 'mlqa_hi_translated': 
            return concatenate_datasets([
                self.load_splits('mlqa', 'mlqa-translate-train.hi'), 
                self.load_splits('mlqa', 'mlqa-translate-test.hi'),                     
            ])
        elif dataset_name == 'mlqa_hi_full': 
            return concatenate_datasets([
                self.load_splits('mlqa', config) for config in self.mlqa_hi_full_configs
            ])
        elif dataset_name == 'xquad': 
            return concatenate_datasets([
                self.load_splits('xquad', 'xquad.hi'),
                self.load_splits('xquad', 'xquad.en'), 
            ])
        elif dataset_name == 'mlqa_hi_en': 
            MLQA_HI_AND_EN = ['mlqa.hi.hi', 'mlqa.hi.en', 'mlqa.en.hi']
            return concatenate_datasets([self.load_splits('mlqa', config) for config in MLQA_HI_AND_EN])
        
        elif dataset_name == 'squad_ta_3k': 
            df = pd.read_csv(self.df_dir/'squad_tamilQA.csv')
            df['id'] = df.index 
            df.id = df.id.apply(str)
            return self.df_to_dataset(df)
        else: 
            raise Exception(f'{dataset_name} not found')
    
    def load_splits(self, dataset_name, config=None):
        dataset = datasets.load_dataset(dataset_name, config)
        splits = []
        for split in ['train', 'validation', 'test']: 
            if split in dataset: 
                splits.append(dataset[split])
        return concatenate_datasets(splits)   
    
    def _standardize_df_answers(self, row): 
        if len(row.answer_text) == 0: 
            return {'text': [], 'answer_start': []}
        return {
            'text': [row.answer_text], 
            'answer_start': [row.answer_start], 
        }
    
    def df_to_dataset(self, df): 
        df['answers'] = df.apply(self._standardize_df_answers, axis=1)
        dataset = datasets.Dataset.from_pandas(df)
        return dataset

    # Info About the Datasets
    def random_example(self, dataset): 
        i = random.randint(0, len(dataset))
        ex = dataset[i]
        q, c, a = ex['question'], ex['context'], ex['answers']['text']
        print(f'question ({len(q)} chars): {q}')
        print(f'answer: {a}')
        print(f'context ({len(c)} chars): {c}')
        print()

    def _get_ans_len(self, example): 
        answers = example['answers']
        if isinstance(answers, str): return len(answers)
        if len(answers['text']) == 0: return 0
        return len(answers['text'][0])

    def _avg_len(self, dataset, col_name): 
        return sum([len(ex) for ex in dataset[col_name]]) / len(dataset)    

    def print_dataset_info(self, dataset, name): 
        print()
        print(f'{red(name)} Dataset')
        print(f'Total {name} examples: ', green(len(dataset)))
        print(f'Average answer length: ', blue(sum(self._get_ans_len(ex) for ex in dataset)/len(dataset)))
        print(f'Average context length: ', blue(self._avg_len(dataset, 'context')))
        print(f'Average question length: ', blue(self._avg_len(dataset, 'question')))
        print('----------------------------')
        self.random_example(dataset)
        


    @staticmethod
    def prepare_train_features(examples, tokenizer, max_seq_len, doc_stride):
        # Get rid of surrounding whitespace
        examples['question'] = [q.strip() for q in examples['question']] 
        
        # Toeknize examples with truncation/padding on context, but keep overflows with doc stride
        # Get multiple features from a single examples
        tokenized_examples = tokenizer(
            examples['question'], 
            examples['context'], 
            truncation='only_second', 
            max_length=max_seq_len, 
            stride=doc_stride, 
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length', 
        )
        
        # Feature to it's corrosponding example
        feature_to_example_idx = tokenized_examples.pop('overflow_to_sample_mapping')
        
        # Token to char position in the original context
        # Used to compute start_positions, end_positions
        offset_mapping = tokenized_examples.pop('offset_mapping')
        
        # Start positions, end positions are calculated
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for batch_index, offsets in enumerate(offset_mapping): 
            # offsets contain offset mapping for tokens in the feature
            
            input_ids = tokenized_examples["input_ids"][batch_index]
            cls_index = input_ids.index(tokenizer.cls_token_id) # mostly 0
            
            # get the sequence for full context and question
            tokens_to_sentence_id = tokenized_examples.sequence_ids(batch_index)
            
            # index of example that contains this span
            example_idx = feature_to_example_idx[batch_index]
            answers = examples['answers'][example_idx]
            
            no_answers = len(answers['text']) == 0 
            if no_answers: 
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else: 
                # Start/end character index of the answer in the text (Only takes first answer)
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                # Start token index of the current span in the text.
                token_start_index = 0
                while tokens_to_sentence_id[token_start_index] != 1:
                    token_start_index += 1
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while tokens_to_sentence_id[token_end_index] != 1:
                    token_end_index -= 1
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    @staticmethod
    def prepare_valid_features(examples, tokenizer, max_seq_len, doc_stride):
        examples['question'] = [q.strip() for q in examples['question']]
        tokenized_examples = tokenizer(
            examples['question'], 
            examples['context'], 
            truncation='only_second',  
            max_length=max_seq_len, 
            stride=doc_stride, 
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length', 
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples


class TFDataModule: 
    """
    Convert huggingface dataset to tensorflow dataset 
    
    TODO
    - Add TFRecord support in future for other competitions 
    - How to add more flexibility in model inputs / outputs ?
    - Am I missing any optimizations?
    - Will caching help in this case ?
    - Go through: https://www.tensorflow.org/guide/data_performance
    - Add tensorboard profiling ? 
    - Seprate class for TFAugment for image competitions ?
    """
    def __init__(
        self, 
        hf_dataset_train, hf_dataset_valid, 
        tensor_keys=['input_ids', 'token_type_ids', 'attention_mask'], 
        label_keys=['start_positions', 'end_positions'], 
    ):
        self.train_samples = len(hf_dataset_train)
        self.valid_samples = len(hf_dataset_valid)
        print('Number of train features: ', self.train_samples)
        print('Number of valid features: ', self.valid_samples)
        
        self.tensor_keys = tensor_keys
        self.label_keys = label_keys
        
        self.train_ds_unbatched = self.huggingface_dataset_to_tfds(hf_dataset_train)
        self.valid_ds_unbatched = self.huggingface_dataset_to_tfds(hf_dataset_valid)
        
    def huggingface_dataset_to_tfds(self, hf_dataset): 
        # HACK: Setting to format to numpy reduces time by 10-100x
        # Warning: Setting type for all numpy arrays as np.int32
        
        hf_dataset.set_format(type='numpy')
        start_time = time()
        features = {
            x: hf_dataset[x].astype(np.int32) for x in self.tensor_keys
        }
        ds = tf.data.Dataset.from_tensor_slices(features)
        
        are_labels_present_in_hf_dataset = all([key in hf_dataset.features for key in self.label_keys])
        if are_labels_present_in_hf_dataset: 
            labels = {
                x: hf_dataset[x].astype(np.int32) for x in self.label_keys
            }
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            ds = tf.data.Dataset.zip((ds, label_ds))
        
        print(green(time()-start_time), f'seconds to build the dataset with {len(hf_dataset)} samples')
        return ds
    
    def prepare_datasets(self, batch_size, buffer_size=4096):
        print('Building datasets with batch size', blue(batch_size))
        train_ds = self.train_ds_unbatched.shuffle(buffer_size, reshuffle_each_iteration=True).repeat().batch(batch_size)
        valid_ds = self.valid_ds_unbatched.batch(batch_size, drop_remainder=True) # drop_remainder to show shape
        
        train_steps, valid_steps = self.train_samples//batch_size+1, self.valid_samples//batch_size
        print('train steps/epoch: ', colored(train_steps, 'blue'))
        print('valid steps/epoch: ', colored(valid_steps, 'blue'))
        
        return train_ds.prefetch(tf.data.AUTOTUNE), train_steps, valid_ds.prefetch(tf.data.AUTOTUNE), valid_steps
    
    
