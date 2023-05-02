import torch
import torch.nn as nn
import numpy as np
from datetime import datetime as dt

class fastGraph(nn.Module):

    def __init__(self, edge_fn):
        super(fastGraph, self).__init__()

        self.edge = edge_fn
        self.l = None
        self.S = None

    def fit(self, Ex, ids, timed=False):
        """

        :param Ex: Vectors used to represent each utterance.
        :param ids: 1:1 label for the example number of each vec in Ex
        :return: None. Creates dynamic socio-semantic graph
        """
        l = np.unique(ids)
        combos = []
        for i, _ in enumerate(l):
            for j, _ in enumerate(l):
                combos += [np.sort([i, j])]
        combos = np.unique(combos, axis=0)

        self.S = torch.zeros(size=(len(l), len(l)))

        if timed:
            start = dt.now()

        for i, j in combos:
            ij, ji = self.edge(Ex[ids == l[i]], Ex[ids == l[j]])
            self.S[i, j] = ij
            self.S[j, i] = ji

        if timed:
            end = dt.now()
            print(end-start)

    def fit_with_indexing(self, Ex, ids, indexes, timed=False):
        """

        :param Ex: Vectors used to represent each utterance.
        :param ids: 1:1 label for the example number of each vec in Ex
        :return: None. Creates dynamic socio-semantic graph
        """
        l = np.unique(ids)
        combos = []
        for i, _ in enumerate(l):
            for j, _ in enumerate(l):
                combos += [np.sort([i, j])]
        combos = np.unique(combos, axis=0)

        self.S = torch.zeros(size=(len(l), len(l)))

        if timed:
            start = dt.now()

        for i, j in combos:
            ij, ji = self.edge.on_indexes(Ex[ids == l[i]], Ex[ids == l[j]], indexes[ids == l[i]], indexes[ids == l[j]])
            self.S[i, j] = ij
            self.S[j, i] = ji

        if timed:
            end = dt.now()
            print(end-start)

    def __getitem__(self, item):
        return self.S[item]