from termcolor import colored
from time import time 
import pandas as pd

class CompetitionDataModule: 
    """
    Wrapper class for competition related QA dataframes
    - Read, clean & process competition dataframes
    - Add special tokens, transform into goldP + uniform negative 
    """
    def __init__(
        self, df_dir, fold, 
        word_counts=None, context_lengths=None, 
        word_count_tokens_position=None, context_length_tokens_position=None, 
    ): 
        self.df_dir = df_dir 
        self.fold = fold
        
        self.word_counts = word_counts 
        self.context_lengths = context_lengths 
        
        self.word_count_tokens_position = word_count_tokens_position
        self.context_length_tokens_position = context_length_tokens_position
    
    
    # --- PUBLIC API ---
    def prepare_original_dataframe(
        self, transformation='goldp', min_paragraph_len=64, 
    ): 
        start_time = time()
        df = pd.read_csv(self.df_dir/'comp_org.csv')
        df = self._clean_kaggle_noise(df)
        df = self.add_special_tokens(df)
        train, valid = df[df.fold!=self.fold], df[df.fold==self.fold]
        if transformation == 'goldp': 
            print('Making a goldp and a uniform negative dataframe from original dataframe')
            gold = self.convert_to_goldp(train)
            negative = self.convert_to_uniform_negative(train, min_paragraph_len)
            train = pd.concat([gold, negative])
        
        print(colored(time()-start_time, 'blue'), 'seconds to prepare original dataframe')
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