import pandas as pd
import numpy as np
import regex as re
from torch.utils.data import Dataset

class data(Dataset):

    def __init__(self, data_path, tokenizer, add_special_tokens=False, w_column=None, text_column=None):
        super(data, self).__init__()
        self.df = pd.read_csv(data_path)
        self.sptoks = add_special_tokens
        self.tok = tokenizer
        self.W_ = w_column
        self.X_ = text_column

    def __len__(self):
        return len(self.df)

    def delta(self, w_, sent):
        W = re.compile(r'\b' + re.escape(w_) + r'\b')
        locs = [i.start() for i in re.finditer(W, sent)]
        idx = [len(self.tok.encode(sent[:i], add_special_tokens=self.sptoks))-1 if self.sptoks == True
               else len(self.tok.encode(sent[:i], add_special_tokens=self.sptoks))
               for i in locs]
        return idx

    def __getitem__(self, item):
        return self.df[self.X_].loc[item].values.tolist(), [self.delta(self.df[self.W_].loc[idx], self.df[self.X_].loc[idx]) for idx in item]
