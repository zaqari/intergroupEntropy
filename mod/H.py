import torch
import torch.nn as nn
import pandas as pd
import numpy as np

cos = nn.CosineSimilarity(dim=-1)
s_col = 'subreddit_name'

def vec_from_string(dfi):
    return torch.FloatTensor(
        [
            [
                float(i) for i in vec.replace('[', '').replace(']','').split(', ')
            ] for vec in dfi['vecs'].values
        ]
    )

groups = ['menslib', 'feminism', 'mensrights']

location_path = "/home/zaq/d/convergence/feminism-menslib-mensrights/roe/"
data_name = "vecs.tsv"

output_path = location_path + "posteriors.pt"
H_summary_path = location_path + "summaryH.csv"
TTest_summary_path = location_path + "TTest.csv"

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

Eu = vec_from_string(df)
del df['vecs']
print(Eu.shape)

#Establish parameters for Gaussian
d = torch.distributions.Normal(1,.3,validate_args=False)

def P(x):
    return torch.exp(d.log_prob(x))

# (1) monte carlo procedure
# N_permutations, xsize, ysize = 1000, 20, 20
N_permutations, xsize, ysize = 200, 200, 100

sample_sets = np.array([[j for j in np.unique(ids) if j != i] for i in np.unique(ids)])
comm_status = comm[sample_sets]
comment_n = commN[sample_sets]

ids = df['__id'].values

M = []
for permutation in range(N_permutations):

    #dictionary of sampled comments of shape group x sample_size
    xm = {group: np.random.choice(np.unique(ids)[comm==group], size=(xsize,), replace=False) for group in groups}

    #dictionary of indeces for each sample per group of shape group x total_number_of_vecs
    idxm = {group: df['__id'].isin(xm[group]).values for group in groups}

    #dictionary of masks for each sample group for summation of shape group x sample_size x total_number_of_vecs
    mxm = {group: torch.FloatTensor([df['__id'].values[idxm[group]]==idx for idx in xm[group]]) for group in groups}

    #dictionary of y-axis samples for each sampled conversation of shape sample_conversation x group x y_axis_sample_size
    ysamples = {idx:
                    { group:
                          np.random.choice(sample_sets[idx][(comm_status[idx] == group) & (comment_n[idx] != commN[idx])], size=(ysize,), replace=False)
                      for group in groups }
        for idx in sum([xm[group].tolist() for group in groups],[])}

    key = list(ysamples.keys())[0]

    #dictionary of y-axis sample indeces for each sampled conversation of shape sample_conversation x group x total_number_of_vecs
    ysamples = { k:
                    { group:
                          df['__id'].isin(v[group]).values
                      for group in groups }
                for k,v in ysamples.items() }

    m = []
    for group in groups:
        #calculating response per each group of shape idxm[group] x ysamples[group,sample_conversation,groupY] and taking the max for each.
        r = torch.cat([
            torch.cat([
                cos(
                    Eu[ids==x].unsqueeze(1),
                    Eu[ysamples[x][groupy]]).max(dim=-1).values.unsqueeze(-1)
                for groupy in groups], dim=-1)
            for x in xm[group]],dim=0)

        r = P(r)
        r = -(r * torch.log(r))
        r = (mxm[group].unsqueeze(-1) * r).sum(dim=1)
        m += [r.unsqueeze(0)]

    m = torch.cat(m, dim=0)
    M += [m.unsqueeze(0)]

    if ((permutation+1) % 10) == 0:
        print('permutation {}/{}'.format(permutation+1, N_permutations))

# M is a matrix of shape trials x groups x sample_size x group_history_sampled_from
M = torch.cat(M,dim=0)
torch.save({'M':M}, output_path)

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
