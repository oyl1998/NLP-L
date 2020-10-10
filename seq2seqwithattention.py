# -*- coding: utf-8 -*-
'''
Name: seq2seqwithattention.py
Auth: long_ouyang
Time: 2020/10/9 21:58
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the German and English spaCy models
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout) # Output is of the same shape as input

    def forward(self, src):
        # src = [src_len, batch_size]
        src = src.transpose(0, 1) # sec = [batch_Size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1) # embedded = [src_len, batch_size, emb_dim]
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)
        # enc_hidden is [forward_1, backward_1, forward_2, backward_2, ···,]
        # enc_output are always from the last layers
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        return enc_output, s

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False) # 这个dec_hid_dim就是？可以自己设定

    def forward(self, s, enc_output):
        # s = [batch_Size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2) # 去除第2个维度

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_Size, enc_hid_dim * 2]
        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1) # 在第1维上增加

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        c = torch.bmm(a, enc_output).transpose(0, 1) # torch.bmm 矩阵乘法，a and enc_output size is 3 necessarily

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len, batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_direction, batch_size, dec_hid_dim]
        dec_ouput, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_ouput = dec_ouput.squeeze(0)
        c = c.squeeze(0)

        y_pred = self.fc_out(torch.cat((dec_ouput, c, embedded), dim=1))

        return y_pred, dec_hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device # cuda or cpu

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is ??? teacher_forcing
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequeeze, forward and backward
        enc_output, s = self.encoder(src)

        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return outputs

