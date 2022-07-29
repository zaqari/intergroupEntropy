import torch
import torch.nn as nn

class entropy(nn.Module):
    
    def __init__(self, sigma=.3):
        super(entropy, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.N = torch.distributions.Normal(1,scale=sigma,validate_args=False)
        self.side = 'right'

    def forward(self, ex, ey, side=None):
        if bool(side):
            self.side=side

        C = self.cos(ex.unsqueeze(1), ey)

        if self.side == 'right':
            C = self.N.log_prob(C.max(dim=-1).values)
            return (torch.exp(C) * C).sum()

        elif self.side == 'left':
            C = self.N.log_prob(C.max(dim=0).values)
            return (torch.exp(C) * C).sum()

        else:
            C1, C2 = C.max(dim=-1), C.max(dim=0)
            return (torch.exp(C1) * C1).sum(), (torch.exp(C2) * C2).sum()

