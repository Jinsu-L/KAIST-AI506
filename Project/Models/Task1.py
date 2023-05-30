import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.GAT import GAT
from dgl.nn import GraphConv
from dgl.nn import GATConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Task1Net(nn.Module):
    def __init__(self, n_itemset, n_users, user_embedding, item_embedding, style_embedding):
        super().__init__()

        self.itemset_embedding = nn.Embedding(n_itemset, embedding_dim=64)
        # self.user_embedding = nn.Embedding(n_users, embedding_dim=256)
        self.user_embedding = user_embedding # nn.Embedding(n_users, embedding_dim=64)
        self.item_embedding = item_embedding # nn.Embedding(n_users, embedding_dim=64)
        self.style_embedding = style_embedding # nn.Embedding(n_users, embedding_dim=64)

        self.mix_lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, proj_size=128, batch_first=True,
                                bidirectional=True, dropout=0.2)
        self.lstm_mix = nn.Linear(256, 128)

        self.fc = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)


    def forward(self, user, itemset, item_list=[], length_list=[]):
        user_embedding = self.user_embedding(user)
        itemset_embedding = self.itemset_embedding(itemset)

        # 나중에 item embeding 만들면 input으로 받아서,
        # lstm으로 mixing. 각 itemset 별로 item의 개수가 다 다름..

        h = torch.concat([user_embedding, itemset_embedding], 1)
        h = F.elu(self.fc(h))
        h = F.dropout(h, 0.3)
        y = F.sigmoid(self.fc2(h))

        return y





