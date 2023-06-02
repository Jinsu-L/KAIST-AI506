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

        # Simple NN
        self.itemset_embedding = nn.Embedding(n_itemset, embedding_dim=64)
        self.user_free_embedding = nn.Embedding(n_users, embedding_dim=64)

        # NGCF
        self.user_embedding = user_embedding  # nn.Embedding(n_users, embedding_dim=64)
        self.item_embedding = item_embedding  # nn.Embedding(n_items, embedding_dim=64)

        # CASG
        self.style_embedding = style_embedding  # nn.Embedding(n_items, embedding_dim=64) # sota-144

        # itemset mix
        # 64 + 144
        self.mix_lstm = nn.LSTM(input_size=64+144, hidden_size=256, num_layers=2, proj_size=128, batch_first=True,
                                bidirectional=True, dropout=0.1)

        # self.lstm_mix = nn.Linear(128*2, 128)

        self.fc = nn.Linear(64 * 3 + 256 + 1, 128) # (64*2, 128) # (64 * 3 + 256 + 1, 128) # ?64
        self.fc2 = nn.Linear(128, 1)
        # self.fc3 = nn.Linear(10, 1)

        # self.bn1 = torch.nn.BatchNorm1d(32)
        # self.bn2 = torch.nn.BatchNorm1d(32)

        self.pop_bn = torch.nn.BatchNorm1d(4)
        self.pop_fc = nn.Linear(4, 1)

        # self.pop_out = nn.Linear(2, 1)

        self.bn1 = torch.nn.BatchNorm1d(64 * 3 + 256 + 1) # (64*2) # (64 * 3 + 256 + 1)
        self.bn2 = torch.nn.BatchNorm1d(128)


    def forward(self, user, itemset, item_list=[], length_list=[], pop=[]):
        user_embeds = self.user_embedding[user]
        itemset_embeds = self.itemset_embedding(itemset)
        user_free_embeds = self.user_free_embedding(user)

        # print(item_list.shape, item_list[0], length_list[0])
        # 나중에 item embeding 만들면 input으로 받아서,
        # lstm으로 mixing. 각 itemset 별로 item의 개수가 다 다름..
        item_embeds = self.item_embedding[item_list]  # batch, 5, dim
        style_embeds = self.style_embedding[item_list]  # batch, 5. dim
        concated_item_embeds = torch.cat([item_embeds, style_embeds], 2)  # batch, 5. 2*dim

        # print(concated_item_embeds.shape)
        input_batch = pack_padded_sequence(concated_item_embeds, length_list.tolist(), batch_first=True,
                                           enforce_sorted=False)
        packed_output, _ = self.mix_lstm(input_batch)  # batch, dim * 4
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        itemset_item_embeds = output[range(len(output)), length_list - 1, :]  # batch, dim
        # self.lstm_mix(F.elu(

        pop_score = F.tanh(self.pop_fc(self.pop_bn(pop)))

        # user_embeds
        # F.relu(
        h = torch.concat([user_embeds, user_free_embeds, itemset_embeds, itemset_item_embeds, pop_score], 1)  # batch, dim*3
        # h = F.relu(torch.concat([user_free_embeds, itemset_embeds, itemset_item_embeds, pop_score], 1))  # batch, dim*3
        # h = F.relu(torch.concat([user_free_embeds, itemset_embeds, pop_score], 1))  # batch, dim*3
        h = self.bn1(h)
        h = F.relu(self.fc(h))
        h = self.bn2(h)
        y = F.sigmoid(self.fc2(h))
        # h = F.dropout(h, 0.2)
        # y = F.sigmoid(self.fc3(h))

        return y
