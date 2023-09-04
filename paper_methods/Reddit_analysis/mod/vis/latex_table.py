import numpy as np
import pandas as pd

def latex_table(df, add_hline=False):
    latex = '\hline \n ' + ' & '.join(list(df)) + '\\\ \n \hline\hline \n '

    for row in df.index:

        latex += ' & '.join(df.loc[row].astype(str).values) + '\\\ \n '
        if add_hline:
            latex += ' \hline \n '

    return latex

def format_num_string(x):
    k = str(x)
    if 'e-01' in k:
        k = k.split('e')[0].replace('.', '')
        k = '.'+k
        k = k.replace('.-', '-.')
    if 'e-02' in k:
        k = k.split('e')[0].replace('.', '')
        k ='.0'+k
        k = k.replace('.0-', '-.0')
    return k

