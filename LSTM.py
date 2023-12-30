import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_embeddings,num_layers=2):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_embeddings = num_embeddings
        self.char_embedding = nn.Embedding(self.input_size,self.num_embeddings)
        self.lstm = nn.LSTM(self.num_embeddings, hidden_size,num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, self.output_size)
        self.init_weight()
        # (n,T,15) output, labels (n,T)
    def forward(self, input):
        # input = self.embedding(input)
        # print(input.shape)
        output = self.char_embedding(input)
        output, _ = self.lstm(output)
        output = self.linear(output)
        return output

    def init_weight(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)