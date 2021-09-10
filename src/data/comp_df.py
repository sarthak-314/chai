import pandas as pd

def fix_start(df): 
    def func(row): 
        return row.context.find(row.answer_text) 
    df['answer_start'] = df.apply(func, axis=1)
    return df

def clean_kaggle_noise(df):
    df = df.set_index('id')
    df.loc['1a2160a69', 'answer_text'] = 'किर्काल्दी'
    df.loc['632c16ba0', 'answer_text'] = 'चार जोड़े'
    df = df[df.index!='2b41f3744']
    df = df[df.index!='bc9f0d533'] 
    df = df.reset_index()
    df = fix_start(df)
    return df

def read_fold(df_dir, fold): 
    df = pd.read_csv(df_dir/'comp_org.csv')
    df = clean_kaggle_noise(df)
    train, valid = df[df.fold!=fold], df[df.fold==fold]
    return train, valid

def build_goldp(df): 
    # Take the first sentence where the answer appears
    gold = {'goldp': [], 'id': [], 'context_start': [], 'context_len': []}
    for i, row in df.iterrows(): 
        word_count = 0
        for line in row.context.split('\n'): 
            if row.answer_text in line: 
                if row.id in gold['id']:
                    continue
                gold['goldp'].append(line)
                gold['id'].append(row.id)
                gold['context_start'].append(word_count)
                gold['context_len'].append(len(row.context))
            word_count += len(line)
    gold = pd.DataFrame(gold)
    gold = df.merge(gold)
    gold['context'] = gold['goldp']
    gold = fix_start(gold)
    return gold

def build_uniform_negative(df, min_paragraph_len=128):
    negative = {'negative_p': [], 'id': []}
    for i, row in df.iterrows(): 
        for line in row.context.split('\n'): 
            if row.answer_text not in line: 
                if len(line) < min_paragraph_len: continue
                negative['negative_p'].append(line)
                negative['id'].append(row.id)

    negative = pd.DataFrame(negative)
    negative = df.merge(negative)
    negative['context'] = negative['negative_p']
    negative['answer_text'] = ''
    negative['answer_start'] = 0

    del negative['context_with_token']
    del negative['org_context']
    del negative['negative_p']
    return negative