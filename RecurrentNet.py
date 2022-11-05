import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RecurrentNet(torch.nn.Module):
    def __init__(self, n_layers, seq_size, dropout, netType, n_inputs = 8, n_outputs = 10):
        super(LSTMNet, self).__init__()
        self.n_layers = n_layers
        self.seq_size = seq_size
        if netType == 'LSTM':
            self.rec = nn.LSTM(n_inputs, seq_size, n_layers, dropout = dropout)
        if newType == 'GRU':
            self.rec = nn.GRU(n_inputs, seq_size, n_layers, dropout = dropout)
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(seq_size, seq_size)
        self.hidden2 = nn.Linear(seq_size, seq_size/2)
        self.output = nn.Linear(seq_size/2, n_outputs)
       
        
    def forward(self, x):
        out, _ = self.rec(x) 
        out = self.relu(self.hidden1(out))
        out = self.relu(self.hidden2(out))
        out = self.output(out) 
        return out
        