import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import controlledMaskEmbeddedDropout
from locked_dropout import controlledMaskLockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.dropi = controlledMaskLockedDropout(dropouti)
        self.droph = torch.nn.ModuleList([controlledMaskLockedDropout(dropouth) for l in range(nlayers-1)])
        self.drop = controlledMaskLockedDropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.embedded_dropout = controlledMaskEmbeddedDropout(self.encoder, dropoute)
        assert rnn_type in ['LSTM', 'QRNN'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else ninp, save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight_raw

        self.W = nn.Linear(ninp, ninp)
            
        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight_raw.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, draw_mask=True, emb=None):
        if emb is None:
            emb = self.embedded_dropout(draw_mask, input)

        raw_output = self.dropi(draw_mask, emb)

        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(draw_mask, raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.droph[l](draw_mask, raw_output)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.drop(draw_mask, raw_output)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        
        Wh = self.W(raw_output)
        
        if return_h:
            return result, hidden, raw_outputs, outputs, emb, Wh
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_())
                    for l in range(self.nlayers)]
