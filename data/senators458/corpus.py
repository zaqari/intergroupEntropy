import pandas as pd
import numpy as np

df = pd.read_csv("/Volumes/V'GER/comp_ling/DataScideProjects/convergenceEntropy/data/senators458/538senators.csv", encoding='ISO-8859-1')
search = ['guns', 'gun control', 'gun rights', '2nd amendment', 'rifle']
indeces = [i for i in df.index if sum([wi in str(df['text'].loc[i]).lower() for wi in search]) > 0]

df = df.loc[indeces].copy()
df.index=range(len(df))

print(df['party'].value_counts())
df.columns = ['created_at', 'comment']+list(df)[2:]
df['comment'] = [comment.replace('\n', ' ') for comment in df['comment'].values]
df.to_csv("/Volumes/V'GER/comp_ling/DataScideProjects/convergenceEntropy/data/senators458/twitter_gun-control.csv", index=False)