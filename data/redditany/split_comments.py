import pandas as pd
import numpy as np

title_vals = {
    'Dr.': 'Dr',
    'Mr.': 'Mr',
    'Mrs.': 'Mrs',
    'Ms.': 'Ms',
    'Sr.': 'Sr',
    'Jr.': 'Jr',
    'dr.': 'Dr',
    'mr.': 'Mr',
    'mrs.': 'Mrs',
    'ms.': 'Ms',
    'sr.': 'Sr',
    'jr.': 'Jr',
}

split_vals = {
    '.': '.<sep>',
    '?': '?<sep>',
    '!': '!<sep>',
}

def split_values(df, col):
    meta_data_cols = [c for c in list(df) if c != col]
    dfi = []
    for j in df.index:
        meta_data = df[meta_data_cols].loc[j].values.tolist()
        text = df[col].loc[j]

        while '\n' in text:
            text = text.replace('\n', ' ')
        while '. .' in text:
            text = text.replace('. .', '..')

        for k,v in title_vals.items():
            text = text.replace(k,v)
        for k,v in split_vals.items():
            text = text.replace(k,v)

        text = text.replace('\t', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        sents = [sent for sent in text.split('<sep>')]
        dfi+=[meta_data+[j, sent] for sent in sents]

    return pd.DataFrame(np.array(dfi), columns=meta_data_cols+['index', col])

