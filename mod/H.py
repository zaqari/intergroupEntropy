import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from kgen2.LM.mutual_information.entropy import entropy


###################################################################################################
### Path variables for project on GPU server
###################################################################################################
location_path = "/home/zaq/d/convergence/feminism-menslib-mensrights/women/"
data_name = "vecs.tsv"
output_path = location_path + "posteriors.pt"
H_summary_path = location_path + "summaryH.csv"
TTest_summary_path = location_path + "TTest.csv"
sample_history_path = location_path + "sample_history.csv"



###################################################################################################
### Import data for analysis
###################################################################################################
s_col = 'subreddit_name'
groups = ['menslib', 'feminism', 'mensrights']

df = pd.read_table(location_path+data_name)
print(list(df))
print(df['subreddit_name'].value_counts())
df = df.loc[
    df[s_col].isin(groups)
    # & df['pattern_found'].values
]
df.index=range(len(df))
df['--id'] = df['__id'].values

update_id_dic = {i:idx for idx,i in enumerate(np.unique(df['__id'].values))}
df['__id'] = df['__id'].replace(update_id_dic)

ids = np.unique(df['__id'].values)
comm = np.array([df[s_col].loc[df['__id'].isin([idx])].unique()[0] for idx in ids])
commN = np.array([df['index'].loc[df['__id'].isin([idx])].unique()[0] for idx in ids])

print(list(df))
for subreddit in df['subreddit_name'].unique():
    print('{} \t {}'.format(subreddit, len(df['__id'].loc[df['subreddit_name'].isin([subreddit])].unique())))



###################################################################################################
### Set up entropy() class
###################################################################################################
H = entropy(sigma=.3, dim=-1).to('cuda')



###################################################################################################
### Create vectors from saved word-vector data
###################################################################################################
def vec_from_string(dfi):
    return torch.FloatTensor(
        [
            [
                float(i) for i in vec.replace('[', '').replace(']','').split(', ')
            ] for vec in dfi['vecs'].values
        ]
    )

Eu = vec_from_string(df)
del df['vecs']
print(Eu.shape)



###################################################################################################
### Set up sampling history location
###################################################################################################
dfsamples = pd.DataFrame(columns=['permutation', 'group', 'xi']+[group+'_ids' for group in groups])
dfsamples.to_csv(sample_history_path,index=False, encoding='utf-8')



###################################################################################################
### Monte Carlo Procedure
###################################################################################################
# (1) monte carlo procedure
# N_permutations, xsize, ysize = 200, 200, 100
N_permutations, xsize, ysize = 500, 200, 50

sample_sets = np.array([[j for j in np.unique(ids) if j != i] for i in np.unique(ids)])
comm_status = comm[sample_sets]
comment_n = commN[sample_sets]

ids = df['__id'].values

M = []
for permutation in range(N_permutations):
    gM = []
    for groupX in groups:
        x = np.random.choice(df['__id'].loc[df['subreddit_name'].isin([groupX])].unique(), size=(xsize,), replace=False)

        m = []
        for xi in x:
            _x = df['__id'].isin([xi]).values

            mi,samples = [],[]
            for groupY in groups:
                y = np.random.choice(df['__id'].loc[
                                          df['subreddit_name'].isin([groupY])
                                          & ~_x
                                          ].unique(), size=(ysize,), replace=False)
                _y = df['__id'].isin(y).values

                mi += [H(Eu[_x].to('cuda'), Eu[_y].to('cuda')).view(1,-1).detach().cpu()]
                samples += [str(y.tolist()).replace(',', '')]

            m += [torch.cat(mi,dim=-1)]

            samples = np.array([permutation, groupX, xi]+samples,dtype='object').reshape(1,-1)
            pd.DataFrame(samples, columns=list(dfsamples)).to_csv(sample_history_path, header=False, index=False, encoding='utf-8', mode='a')

        gM += [torch.cat(m,dim=0).unsqueeze(0)]

    M += [torch.cat(gM,dim=0).unsqueeze(0)]

    if ((permutation+1) % 10) == 0:
        print('permutation {}/{}'.format(permutation+1, N_permutations))

# M is a matrix of shape trials x groups x sample_size x group_history_sampled_from
M = torch.cat(M,dim=0)
torch.save({'M':M}, output_path)



###################################################################################################
### Analysis of sampling outputs
###################################################################################################
Hdata = []
for i, group in enumerate(groups):
    for j,comparison in enumerate(groups):
        res = M[:,i,:,j].reshape(-1)
        Hdata += [[group+':'+comparison,res.mean().item(),res.median().item()]]
Hdata = pd.DataFrame(np.array(Hdata), columns=['comparison', 'mean', 'median'])
Hdata.to_csv(H_summary_path, encoding='utf-8')


from scipy.stats import ttest_ind as ttest
Tdata = []
for i, group in enumerate(groups):
    for j,comparison in enumerate(groups):
        if group != comparison:
            sample1 = M[:,i,:,i].reshape(-1).numpy()
            sample2 = M[:,i,:,j].reshape(-1).numpy()
            test_results = ttest(sample1,sample2)
            Tdata += [['(ttest) {}:{}'.format(group, comparison), str([test_results.statistic, test_results.pvalue])]]
Tdata = pd.DataFrame(np.array(Tdata), columns=['cond', 'test'])
Tdata.to_csv(TTest_summary_path, encoding='utf-8')
