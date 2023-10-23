import torch
import numpy as np
import torch.nn as nn

class entropy(nn.Module):
    
    def __init__(self, sigma=.3, dim=None):
        super(entropy, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.N = torch.distributions.Normal(1,scale=sigma, validate_args=False)
        self.dim = dim

    def forward(self, ex, ey, dim=None):
        if bool(dim):
            self.dim=dim

        # Get cosine similarity comparison between lexical items
        C = self.cos(ex.unsqueeze(1), ey)

        if self.dim != None:
            #along a single dimension,
            # (1) Get max cosine similarity
            # (2) Get log prob of similarity
            # (3) Calculate log prob and entropy
            C = self.N.log_prob(1+C.max(dim=self.dim).values)
            return -(torch.exp(C) * C).sum()

        else:
            C1, C2 = self.N.log_prob(1+C.max(dim=-1).values), self.N.log_prob(1+C.max(dim=0).values)
            C = None
            return -(torch.exp(C1) * C1).sum(), -(torch.exp(C2) * C2).sum()

    def unsummed(self,ex,ey,dim=None):
        if bool(dim):
            self.dim=dim

        C = self.cos(ex.unsqueeze(1), ey)

        C = self.N.log_prob(1+C.max(dim=self.dim).values)
        return -(torch.exp(C) * C)

    def on_indexes(self, ex, ey, x_indeces, y_indeces):
        C = self.cos(ex.unsqueeze(1), ey)

        C1, C2 = self.N.log_prob(1+C.max(dim=-1).values[x_indeces]), self.N.log_prob(1+C.max(dim=0).values[y_indeces])
        return -(torch.exp(C1) * C1).sum(), -(torch.exp(C2) * C2).sum()

class informativeness(nn.Module):

    def __init__(self, sigma=.3, dim=None):
        super(informativeness, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.N = torch.distributions.Normal(1, scale=sigma, validate_args=False)
        self.dim = dim

    def forward(self, ex, ey, dim=None):
        if bool(dim):
            self.dim = dim

        # Get cosine similarity comparison between lexical items
        C = self.cos(ex.unsqueeze(1), ey)

        if self.dim != None:
            # along a single dimension,
            # (1) Get max cosine similarity
            # (2) Get log prob of similarity
            # (3) Calculate log prob and entropy
            C = self.N.log_prob(1+C.max(dim=self.dim).values)
            return -C.sum()

        else:
            C1, C2 = self.N.log_prob(1+C.max(dim=-1).values), self.N.log_prob(1+C.max(dim=0).values)
            C = None
            return -C1.sum(), -C2.sum()

    def unsummed(self, ex, ey, dim=None):
        if bool(dim):
            self.dim = dim

        C = self.cos(ex.unsqueeze(1), ey)

        return self.N.log_prob(C.max(dim=self.dim).values)

    def on_indexes(self, ex, ey, x_indeces, y_indeces):
        C = self.cos(ex.unsqueeze(1), ey)

        C1, C2 = self.N.log_prob(1+C.max(dim=-1).values[x_indeces]), self.N.log_prob(1+C.max(dim=0).values[y_indeces])
        return -C1.sum(), -C2.sum()

class cudaH(nn.Module):

    def __init__(self, sigma=.8, dim=None, stream_at=50):
        super(cudaH, self).__init__()
        self.sigma = sigma
        self.cos = nn.CosineSimilarity(dim=-1).cuda()

        self.N = torch.distributions.HalfNormal(scale=torch.FloatTensor([sigma]).cuda(), validate_args=False)
        # self.dev = device
        self.dim = dim
        self.stream_at = stream_at
        self.stream_n = 2

    def P(self, x):
        return torch.exp(x) / (self.sigma)

    def H(self, x):
        x_ = (1 - x)/(self.sigma**2)
        return self.H_ * x_

    def streamCOS(self, ex, ey):

        spans = int(len(ex) / self.stream_n)

        starts = [i * self.stream_n for i in range(spans)] + [spans * self.stream_n]
        ends = [(i+1) * self.stream_n for i in range(spans)] + [len(ex)]
        steps = list(zip(starts,ends))

        cosM = self.cos(ex[steps[0][0]:steps[0][1]].unsqueeze(1), ey).max(dim=-1).values.view(-1)
        for start, end in steps[1:]:
            if start != len(ex):
                cosM = torch.cat([
                    cosM,
                    self.cos(ex[start:end].unsqueeze(1), ey).max(dim=-1).values.view(-1)
                ], dim=-1)

        return cosM

    def one_sided(self, ex, ey, dim=None):
        if bool(dim):
            self.dim = dim

        if len(ex) >= self.stream_at:
            C = self.streamCOS(ex, ey)

        else:
            C = self.cos(ex.unsqueeze(1), ey).max(dim=self.dim).values

        C = self.N.log_prob(1 + C)

        return -(self.P(C) * C).sum()

    def dual_sided(self, ex, ey):

        if (len(ex) >= self.stream_at) or (len(ey) >= self.stream_at):
            C1, C2 = self.streamCOS(ex, ey).max(dim=-1).values, self.streamCOS(ey, ex).max(dim=-1).values
            C1, C2 = self.N.log_prob(1+C1), self.N.log_prob(1+C2)

        else:
            C = self.cos(ex.unsqueeze(1), ey)
            C1, C2 = C.max(dim=-1).values, C.max(dim=0).values
            C1, C2 = self.N.log_prob(1 + C1), self.N.log_prob(1 + C2)

        return -(self.P(C1) * C1).sum(), -(self.P(C2) * C2).sum()

    def forward(self, ex, ey):
        if self.dim:
            return self.one_sided(ex, ey)
        else:
            return self.dual_sided(ex, ey)

    def unsummed(self, ex, ey, dim=None):
        if bool(dim):
            self.dim = dim

        C = self.streamCOS(ex, ey)

        C = self.N.log_prob(1 + C)
        return -(torch.exp(C) * C)

    def on_indexes(self, ex, ey, x_indeces, y_indeces):
        C = self.cos(ex.unsqueeze(1), ey)

        C1, C2 = self.N.log_prob(1 + C.max(dim=-1).values[x_indeces]), self.N.log_prob(1 + C.max(dim=0).values[y_indeces])
        del C
        return -(torch.exp(C1) * C1).sum(), -(torch.exp(C2) * C2).sum()
