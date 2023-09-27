from entropy import entropy
from bert import BERT

# from kgen2.LM.mutual_information.entropy import cudaH
# from kgen2.LM.LM.BERT.BERT import BERT
# from kgen2.LM.LM.RoBERTa.RoBERTa import RoBERTa

import pandas as pd
from fastGraph import fastGraph
from scipy.stats import ttest_ind as ttest
import glob
import random

H = entropy()
wv = BERT()

def getTransript(fname, gram=False):

    fl = open(fname,'r')
    txt = fl.read()
    fl.close()

    txt = txt.replace("\n\t",' ')
    txt = txt.split("\n")

    tscpt = []
    gra = []
    ids = []

    for i in range(len(txt)-2):
        # let's get staggered * (lexical) content + grammatical tiers; only if aligned
        if txt[i].startswith('*') and (txt[i+2].startswith('%gra') or not gram):        
            ids.append(txt[i][txt[i].find('*'):txt[i].find(':')]) # get id for speaker        
            txt[i] = txt[i].replace(txt[i][txt[i].find('*'):txt[i].find(':')+1],'') 
            txt[i] = txt[i][0:txt[i].rfind(' ')]
            txt[i+2] = txt[i+2].replace(txt[i+2][txt[i+2].find('%'):txt[i+2].find(':')+1],'')
            tscpt.append(txt[i])
            gra.append(txt[i+2])
    
    # return trans, gra, ids
    return [tscpt,gra,ids]

def procChaFile(fname, outfile, baseline = '', mode = 'a', levels=[7,-1], k=10):

    [tscpt, gra, ids] = getTransript(fname)
    if baseline!='':
        [tscpt_bl, gra_bl, ids_bl] = getTransript(baseline)
        if len(tscpt_bl)<len(tscpt):
            # make tscpt the same size as tscpt_bl by cropping
            tscpt = tscpt[0:(len(tscpt_bl)-1)]

    vecs = []
    vecs_comp = []
    for i in range(len(tscpt)):
        vecs.append(wv(tscpt[i],level=[7,-1])[0])
        if baseline!='':
            vecs_comp.append(wv(tscpt_bl[i],level=levels)[0])

    if baseline=='':
        vecs_comp = vecs

    cols = ['fl','i','j','who_i','who_j','n_i','n_j','txt_i','txt_j', 'gra_i','gra_j','h_1','h_2','baseline']
    df = pd.DataFrame(columns=cols) 

    for i in range(len(tscpt)):
        vecs_i = vecs[i]
        # get number of vectors in vecs_i set as n
        n_i = len(vecs_i)
        
        rg = range(max(i-k,0),min(i+k,len(tscpt)))
        for j in rg:
            vecs_j = vecs_comp[j]
            n_j = len(vecs_j)

            h_val = H(vecs_i, vecs_j)
            df = pd.concat([df, pd.DataFrame([[fname,i,j,ids[i],ids[j],
                n_i,n_j,
                tscpt[i],tscpt[j],gra[i],gra[j],
                float(h_val[0]),float(h_val[1]), baseline]], columns=cols)])            
    df.to_csv(outfile, mode=mode, header=(mode=='w'))

# extract all *.cha files into a `canbc' folder in the main.py's path
# make sure all *.cha are in the `canbc' root, which may require extraction
# like this: mv */*.cha .
# https://ca.talkbank.org/access/CABNC.html
files_temp = glob.glob('canbc/*.cha')

# let's meander through and only grab those of a given size!
files = []
for i in range(0,len(files_temp)):
    [tscpt, gra, ids] = getTransript(files_temp[i])
    if (len(tscpt)>=100 and len(tscpt)<=200):
        longs = 0
        for j in range(len(tscpt)):
            # count the number of spaces in tscpt[j]
            if tscpt[j].count(' ')>=100:
                longs = longs + 1
        if longs == 0:
            files.append(files_temp[i])

print(files)
print(len(files)) # files for processing
lix = len(files)
# 2:49 start

# process the first file observed processes
procChaFile(files[0],'results.csv',mode='w')
for i in range(1,lix):
    print('Observed '+str(i))
    procChaFile(files[i],'results.csv')

# sample a random entry from files and save it as comparison_file but exclude from files the ith entry
for i in range(lix):
    print('Shuffling '+str(i))  
    if i == 0: # handle edges
        rg = files[1:(len(files))]
    elif i == (len(files)-1):
        rg = files[0:(len(files)-1)]
    else:  
        # set rg as array between 0 and i-1 and i+1 and len(files)
        rg = files[0:i]
        rg.extend(files[i+1:len(files)])
    comparison_file = random.choice(rg)    
    procChaFile(files[i],'results.csv',baseline=comparison_file)




