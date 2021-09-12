from pathlib import Path
import tensorflow as tf 
import datetime
import os




MONITOR = 'val_loss'
MODE = 'min'
VERBOSE = 2

common_kwargs = {
    'monitor': MONITOR, 
    'mode': MODE, 
    'verbose': VERBOSE, 
}

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

def tb(tb_dir, train_steps): 
    start_profile_batch = train_steps+10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"
    log_path = tb_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_path, histogram_freq=1, update_freq=20,
        profile_batch=profile_range, 
    )
    return tensorboard_callback

def checkpoint(checkpoint_dir=None):
    # checkpoint_filepath = 'checkpoint-{epoch:02d}-{val_loss:.4f}.h5'
    checkpoint_filepath = 'checkpoint.h5'
    if checkpoint_dir is not None: 
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = checkpoint_dir / checkpoint_filepath
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True, 
        **common_kwargs, 
    )

def early_stop(patience=3):
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        restore_best_weights=True, 
        **common_kwargs,
    )

def reduce_lr_on_plateau(patience): 
    return tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=patience,
        min_delta=0.0001,
        min_lr=0,
        **common_kwargs, 
    )

def time_stop(max_train_hours):
    import tensorflow_addons as tfa 
    return tfa.callbacks.TimeStopping(
        seconds=max_train_hours*3600
    )
    
def tqdm_bar(): 
    import tensorflow_addons as tfa
    return tfa.callbacks.TQDMProgressBar()

def terminate_on_nan(): 
    return tf.keras.callbacks.TerminateOnNaN()

def tensorboard_callback(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir)
    )

def wandb_callback():
    from wandb.keras import WandbCallback
    return WandbCallback(
        monitor='val_loss', 
        verbose=0, mode='auto', save_weights_only=True, 
        log_gradients=True, 
    )

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )
    
    
    
class CompDataModule:
    def __init__(
        self, df_dir, fold, 
        word_lens=None, context_lens=None, 
        split_by='paragraph', 
    ): 
        self.df_dir = df_dir
        self.fold = fold
        
        if word_lens is None: 
            print('Skipping adding word length tokens')
        if context_lens is None: 
            print('Skipping adding word length tokens')
        self.word_lens = word_lens
        self.context_lens = context_lens
        self.train, self.valid = self._read_fold(df_dir, fold)
        self.split_by = split_by
    
    def _clean_kaggle_noise(self, df): 
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
    
    def _read_fold(self, df_dir, fold): 
        df = pd.read_csv(df_dir/'comp_org.csv')
        df = self._clean_kaggle_noise(df)
        train, valid = df[df.fold!=fold], df[df.fold==fold]
        return train, valid
    
    def _closest_lower_number(self, nums, target): 
        'Find the closest number in nums that is lower than target'
        for lower, upper in zip(nums, nums[1:]): 
            if upper > target: 
                return lower
        return nums[-1]
    
    def _add_word_len_token(self, content, word_count):
        if self.word_lens is not None: 
            word_token_num = self._closest_lower_number(self.word_lens, word_count)
            word_token = f'[WORD={word_token_num}]'
            content = word_token + content
        return content
    
    def _add_context_len_token(self, content, context_length): 
        if self.context_lens is not None: 
            context_token_num = self._closest_lower_number(self.context_lens, context_length)
            context_token = f'[CONTEXT={context_token_num}]'
            content = context_token + content                    
        return content
    
    def tokenize_context(self, df, add_to_start_of='sentence'): 
        df_dict = {'tokenized_context': [], 'id': []}
        for i, row in df.iterrows():
            if add_to_start_df is 'paragraph': 
                split_on = '\n'
            else: 
                split_on = '।' if row.language is 'hindi' else '.'
            word_count, contents = 0, []
            for content in row.context.split(split_on): 
                if self.context_lens is not None: 
                    content = self._add_context_len_token(content, len(row.context))
                if self.word_lens is not None:
                    content = self._add_word_len_token(content, word_count)
                word_count += len(content)
                contents.append(content)
            tokenized_context = ''.join(contents)
            df_dict['tokenized_context'] = tokenized_context
            df_dict['id'] = row.id
        df = pd.merge(pd.DataFrame.from_dict(df_dict))
        return df
    
    def convert_to_goldp(self, df): 
        'Only select golden paragraphs from the dataframe'
        gold = {'goldp': [], 'id': []}
        for i, row in df.iterrows(): 
            word_count = 0
            for paragraph in row.context.split('\n'): 
                if row.answer_text not in paragraph:
                    word_count += len(paragraph)
                    continue
                paragraph = self._add_word_len_token(paragraph, word_count)
                paragraph = self._add_context_len_token(paragraph, len(row.context))
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
                paragraph = self._add_word_len_token(paragraph, word_count)
                paragraph = self._add_context_len_token(paragraph, len(row.context))
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