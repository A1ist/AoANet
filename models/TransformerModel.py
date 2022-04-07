import torch.nn as nn
import torch
import copy

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
