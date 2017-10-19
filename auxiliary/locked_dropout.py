import torch
import torch.nn as nn
from torch.autograd import Variable

class controlledMaskLockedDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.mask = None
    
    def forward(self, draw_mask, x):
        if self.training == False:
            return x
        if self.mask is None or draw_mask==True:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
            self.mask = Variable(m, requires_grad=False) / (1 - self.dropout)
            self.mask = self.mask.expand_as(x)
        return self.mask * x