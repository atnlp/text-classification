# encoding: utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
MAX_SEQUENCE_LENGTH = 200
epochs = 10
batch_size = 64


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = HIDDEN_DIM  # 128
        self.embedding_dim = EMBEDDING_DIM  # 128
        self.word_embeddings = nn.Embedding(MAX_NB_WORDS, self.embedding_dim)  # (20000, 128)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 2, batch_first=True)  # (128, 128, 2)
        # self.hidden = self.init_hidden()  # (2, 1, 128)
        self.conv2_drop = nn.Dropout(p=0.2)
        # 最终输出的全连接层
        self.fc1 = nn.Linear(self.embedding_dim, 2)

    '''
        def init_hidden(self):
        # shape(2, 1, 128)
        return torch.zeros(2, batch_size, self.hidden_dim, requires_grad=True), torch.zeros(2, batch_size,
                                                                                            self.hidden_dim,
                                                                                          requires_grad=True)

    '''

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(batch_size, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

        embeds = self.conv2_drop(embeds)

        # lstm_out = self.lstm(embeds, self.hidden)
        lstm_out = self.lstm(embeds)

        # 选取最后一个时间步的结果作为最终的输出
        x = lstm_out[0][:, -1, :]
        x = x.view(batch_size, -1)
        x = self.fc1(x)  # F.relu
        x = F.log_softmax(x)
        return x
