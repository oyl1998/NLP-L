# -*- coding: utf-8 -*-
'''
Name: rnn_attention.py
Auth: long_ouyang
Time: 2020/10/8 10:55
'''

import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        # bidirectional 双向
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(in_features=enc_hid_dim * 2, out_features=dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [ src_len, batch_size ]
        src = src.transpose(0, 1) # src = [batch_size, src_len]
        # embedded = [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        # if h0 is not give, it will be set 0 acquiescently
        enc_ouput, enc_hidden = self.rnn(embedded)
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :]), enc_hidden[-1, :, :]), dim=1))
        return enc_ouput, s

