import numpy as np
import torch
from torch.autograd import Variable

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    org_subsource = source[i:i+seq_len+1]
    data = Variable(org_subsource[:-1], volatile=evaluation)
    target = Variable(org_subsource[1:].view(-1))
    if evaluation:
        return data, target
    perm = torch.Tensor(data.size()).float().bernoulli_(1 - args.pred_prob)
    perm[0] = 1
    if args.cuda:
        perm = perm.cuda()
    return data, target, perm
