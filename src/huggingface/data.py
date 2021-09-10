import datasets
from datasets import concatenate_datasets 

from termcolor import colored
import pandas as pd
import functools 
import random

from utils.core import red, blue, green, yellow

class HFDataModule(datasets.Dataset): 
    def __init__(
        self, external_datasets, train, valid, 
        tokenizer, max_seq_len, doc_stride, 
        add_word_len_tokens, add_context_len_tokens, 
        df_dir, 
    ): 
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride 
        self.add_word_len_tokens = add_word_len_tokens
        self.add_context_len_tokens = add_context_len_tokens

        self.df_dir = df_dir 
        self.num_workers = 4

        # TODO: In case of OOM, process features while raw before df
        self.raw_comp_train_dataset = datasets.Dataset.from_pandas(train)
        self.raw_valid_dataset = datasets.Dataset.from_pandas(valid)

        external_dataset_to_loading_fn = {
            'squad_v2': self.load_squad_v2,
            'mlqa_hi_full': self.load_mlqa_hi_full,
            'tydiqa_goldp': self.load_tydiqa_goldp,
            'mlqa_hi_translated': self.load_mlqa_hi_translated,
            'squad_ta_3k': self.load_squad_ta_3k,
            'mlqa_hi_en': self.load_mlqa_hi_en,
            'xquad': self.load_xquad,
            'adversial_qa': self.load_adversial_qa, 
        }
        train_datasets = []
        for dataset_name in external_datasets:
            load_fn = external_dataset_to_loading_fn[dataset_name]
            train_dataset = self._standardize_dataset(HFDataModule.load_fn())
            self.print_dataset_info(train_dataset, dataset_name)
            train_datasets.append(train_dataset.shuffle()) 

        train_datasets = train_datasets + [self.raw_comp_train_dataset]
        
        self.raw_train_dataset = datasets.concatenate_datasets(train_datasets)
        self.prepare_train_features_fn = functools.partial(
            HFDataModule.prepare_train_features,
            tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, doc_stride=self.doc_stride,
        )
        self.processed_train_dataset = self.raw_train_dataset.map(
            self.prepare_train_features_fn, 
            batched=True, num_proc=self.num_workers, 
            remove_columns=self.raw_train_dataset.column_names, 
        )
        self.print_dataset_info(self.raw_comp_train_dataset, 'Competition Training Dataset')
        self.print_dataset_info(self.raw_train_dataset, 'Full Training Dataset')
        print('Size of processed train dataset: ', colored(len(self.processed_train_dataset)))

        self.prepare_valid_features_fn = functools.partial(
            HFDataModule.prepare_valid_features, 
            tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, 
            doc_stride=self.doc_stride, 
        )
        self.processed_valid_dataset = self.raw_valid_dataset.map(
            self.prepare_valid_features_fn, 
            batched=True, num_proc=self.num_workers, 
            remove_columns=self.raw_valid_dataset.column_names, 
        )
        self.print_dataset_info(self.raw_valid_dataset, 'Validation Dataset')
        print('Size of processed valid dataset: ', colored(len(self.processed_valid_dataset)))
    
    
    def load_mlqa_hi_en(self): 
        MLQA_HI_AND_EN = ['mlqa.hi.hi', 'mlqa.hi.en', 'mlqa.en.hi']
        return concatenate_datasets([self.load_splits('mlqa', config) for config in MLQA_HI_AND_EN])

    
    def load_mlqa_hi_full(self): 
        MLQA_HI_FULL = [
            'mlqa.ar.hi', 'mlqa.vi.hi',  'mlqa.zh.hi', 'mlqa.es.hi', 
            'mlqa.hi.ar', 'mlqa.hi.de', 'mlqa.hi.vi', 'mlqa.hi.zh', 'mlqa.hi.es', 
            'mlqa.en.ar', 'mlqa.en.en',
            'mlqa.hi.hi', 'mlqa.hi.en', 'mlqa.en.hi', 
        ]
        return concatenate_datasets([self.load_splits('mlqa', config) for config in MLQA_HI_FULL])
    
    
    def load_mlqa_hi_translated(self): 
        return concatenate_datasets([
            self.load_splits('mlqa', 'mlqa-translate-train.hi'), 
            self.load_splits('mlqa', 'mlqa-translate-test.hi'), 
        ])
    
    
    def load_xquad(self): 
        print('1190 hindi + 1190 english pairs from XQUAD')
        return concatenate_datasets([
            self.load_splits('xquad', 'xquad.hi'),
            self.load_splits('xquad', 'xquad.en'), 
        ])

    
    def load_tydiqa_goldp(self): 
        return self.load_splits('tydiqa', 'secondary_task')

    
    def load_squad_v2(self):
        print('Loading 80k answerable and 50k unanserable questions from squad') 
        return self.load_splits('squad_v2')

    
    def load_adversial_qa(self): 
        print('Loading 30k difficult questions for the model') 
        return self.load_splits('adversial_qa')

    
    def load_squad_ta_3k(self): 
        df = pd.read_csv(self.df_dir/'squad_tamilQA.csv')
        df['id'] = df.index 
        df.id = df.id.apply(str)
        return datasets.Dataset.from_pandas(df)


    def dataset_to_df(self, dataset): 
        FEATURES = ['id', 'context', 'question', 'answers']
        df = pd.DataFrame.from_dict(dataset.to_dict())
        df = df[FEATURES]
        df.answers = df.answers.apply(self._standardize_answers)
        df['context_len'] = df.context.apply(len)
        df['word_start'] = 0
        return df

    def _load_splits(dataset): 
        return datasets.concatenate_datasets(
            dataset[split] for split in ['train', 'validation', 'test']
        )             

    def _standardize_answers(answers):
        answers = dict(answers)
        return {
            'text': list(answers['text']), 
            'answer_start': list(int(ans) for ans in answers['answer_start']), 
        } 

    def _standardize_dataset(self, dataset): 
        df = self.dataset_to_df(dataset)
        return datasets.Dataset.from_pandas(df)

    
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
        print(f'--- {red(name)} Dataset ---')
        print(f'Total {name} examples: ', green(len(dataset)))
        print(f'Average answer length: ', blue(sum(self._get_ans_len(ex) for ex in dataset)/len(dataset)))
        print(f'Average context length: ', blue(self._avg_len(dataset, 'context')))
        print(f'Average question length: ', blue(self._avg_len(dataset, 'question')))
        print('----------------------------')
        self.random_example(dataset)

    
    def prepare_train_features(self, examples, tokenizer, max_seq_len, doc_stride): 
        print('preparing train features')
        examples['question'] = [q.lstrip() for q in examples['question']]
        tokenized_examples = tokenizer(
            examples['question'], 
            examples['context'], 
            truncation='only_second', # Only context truncated
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

    
    def prepare_valid_features(self, examples, tokenizer, max_seq_len, doc_stride):
        examples['question'] = [q.lstrip() for q in examples['question']]
        tokenized_examples = tokenizer(
            examples['question'], 
            examples['context'], 
            truncation='only_second',  # Only Context
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
