import torch
import torch.nn as nn

class entropy(nn.Module):
    
    def __init__(self, sigma=.3):
        super(entropy, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.N = torch.distributions.Normal(1,scale=sigma,validate_args=False)
        self.condense_side = 'right'

    def forward(self, ex, ey, condense_side=None):
        if bool(side):
            self.condense_side=condense_side

        C = self.cos(ex.unsqueeze(1), ey)

        if self.condense_side == 'right':
            C = self.N.log_prob(C.max(dim=-1).values)
            return -(torch.exp(C) * C).sum()

        elif self.condense_side == 'left':
            C = self.N.log_prob(C.max(dim=0).values)
            return -(torch.exp(C) * C).sum()

        else:
            C1, C2 = self.N.log_prob(C.max(dim=-1)), self.N.log_prob(C.max(dim=0))
            return -(torch.exp(C1) * C1).sum(), -(torch.exp(C2) * C2).sum()

