import torch
import torch.nn as nn
import time

class hRQA(nn.Module):

    def __init__(self, sigma=.3, phi=.001, time_execution=False):
        super(hRQA, self).__init__()
        self.delta = nn.Threshold(phi,0.)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.d = torch.distributions.Normal(loc=1, scale=sigma, validate_args=False)
        self.timed = time_execution
        self.M = None

    def streamCOS(self,x,y):
        self.M = torch.cat([self.cos(xi,y).view(1,-1) for xi in x], dim=0)

    def stream(self,Ex,Ey):
        M = self.streamCOS(Ex,Ey)
        M = self.d.log_prob(M)

        return torch.exp(M) * M

    def coe(self,x,y):
        return self.cos(x.unsqueeze(1),y)

    def forward(self,Ex,Ey):
        """
        A model of semantic similarity based on entropy between two utterances of varying
        lengths. The basic intuition can be summarized as follows: Given two utterances,
        the semantic similarity of x to y is based on whether or not you could recover
        the meaning of x from the assorted lexical items in y. To do this, we can calculate
        the recoverability of x from y as follows:

                H(x;y) = Σi maxj( P(xi|yj) log P(xi|yj) )

          (see Coco & Dale 2014; "Cross-recurrence quantification analysis of categorical
          and continuous time series: an R package")

        :param x: a set of vectors for all tokens in sentence x (Ex)
        :param y: a set of vectors for all tokens in sentence y (Ey)
        :return: percent similarity of y to x via shorthand version of Determinism
        """

        C = -self.unspooled_(Ex,Ey)

        return C.min(dim=-1)[0] #self.delta(C).sum(dim=0)

    def unspooled_(self,Ex, Ey):
        C = self.d.log_prob(self.coe(Ex, Ey))

        return torch.exp(C) * C

    def _unspooled_(self,Ex, Ey):
        C = torch.exp(self.d.log_prob(self.coe(Ex, Ey))) / \
            torch.exp(self.d.log_prob(torch.FloatTensor([1])))

        return C * torch.log(1/C)

    def batched(self, Ex, Ey):
        0
