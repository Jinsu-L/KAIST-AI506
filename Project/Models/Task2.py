import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.GAT import GAT
from dgl.nn import GraphConv
from dgl.nn import GATConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Task2Net(nn.Module):
    def __init__(self, g):
        super().__init__()
        # self.embedding = nn.Embedding(len(g.nodes()), embedding_dim=64)
        self.node_feature = g.ndata['feature']
        # self.embed = torch.normal(mean=torch.tensor([[0 for j in range(4)] for i in range(len(g.nodes()))], dtype=torch.float32), std=torch.tensor(1)).to("cuda:0")
        # self.GAT = GAT(g, in_dim=64, hidden_dim=32, out_dim=64, num_heads=8)
        
        self.GConv1_embedding = 32
        self.GConv2_embedding = 32
        self.GConv3_embedding = 48
        self.GConv4_embedding = 32
        self.GConv5_embedding = 32
        self.GConv_output = 64
        self.feature_fc = nn.Linear(self.node_feature.shape[1], self.GConv1_embedding)
        # self.feature_fc2 = nn.Linear(64, 8)

        self.GConv1 = GraphConv(self.GConv1_embedding, self.GConv2_embedding, norm='both', weight=True, bias=True)
        self.GConv2 = GraphConv(self.GConv2_embedding, self.GConv3_embedding, norm='both', weight=True, bias=True)
        self.GConv3 = GraphConv(self.GConv3_embedding, self.GConv_output, norm='both', weight=True, bias=True)
        # self.GConv3 = GraphConv(self.GConv3_embedding, self.GConv4_embedding, norm='both', weight=True, bias=True)
        # self.GConv4 = GraphConv(self.GConv4_embedding, self.GConv5_embedding, norm='both', weight=True, bias=True)
        # self.GConv5 = GraphConv(self.GConv5_embedding, self.GConv_output, norm='both', weight=True, bias=True)
        # self.GConv1 = GATConv(self.GConv1_embedding, self.GConv2_embedding//4, num_heads=4)
        # self.GConv2 = GATConv(self.GConv2_embedding, self.GConv3_embedding//4, num_heads=4)
        # self.GConv3 = GATConv(self.GConv3_embedding, self.GConv_output//4, num_heads=4)
        # self.GConv4 = GATConv(16*4, 16, num_heads=4)
        # self.GConv5 = GATConv(16*4, 16, num_heads=4)
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.5)
        
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)

        # self.g = g
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.fc = nn.Linear(64 * 3, 128, bias=False)
        # self.q_mix = nn.Linear(64*3, 64*3)
        # self.i_mix2 = nn.Linear(128, 4) # 각 아이템 축소용...
        # self.ranker = nn.Linear(128 + len(g.nodes())*4, 32)
        # self.ranker_out = nn.Linear(32, len(g.nodes()))
        # self.q_fc = nn.Linear(128, 64)
        # self.i_fc = nn.Linear(128, 64)

        self.mix_lstm = nn.LSTM(input_size=144, hidden_size=144, batch_first=True, bidirectional=True, proj_size=72)
        # self.lstm_mix = nn.Linear(128, 64)

    def rank(self, query_embedding, all_item_embedding):
        return self.ranker(query_embedding, all_item_embedding)

    def forward(self, g, queries, pos_items, neg_items, query_items, lengths):
        # F.tanh(self.feature_fc2(
        f = F.elu(self.feature_fc(self.node_feature))
        # item_embeds = self.GAT(self.embedding.weight)
        # torch.ones_like(
        # torch.ones_like(self.embedding.weight)
        item_embeds_1 = self.dropout1(F.elu(self.GConv1(g, f))) # edge_weight=g.edata['w']
        # item_embeds_1 = torch.flatten(self.GConv1(g, self.embedding.weight), 1)
        # h = F.elu(item_embeds_1)
        item_embeds_2 = self.dropout2(F.elu(self.GConv2(g, item_embeds_1))) #  edge_weight=g.edata['w']
        # item_embeds_2 = torch.flatten(self.GConv2(g, h), 1)
        # h = F.elu(item_embeds_2)
        item_embeds_3 = self.dropout3(F.elu(self.GConv3(g, item_embeds_2))) # edge_weight=g.edata['w']
        # item_embeds_3 = torch.flatten(self.GConv3(g, h), 1)
        # h = F.elu(item_embeds_3)
        # item_embeds_4 = self.dropout4(F.elu(self.GConv4(g, item_embeds_3, edge_weight=g.edata['w'])))
        # item_embeds_4 = torch.flatten(self.GConv4(g, h), 1)
        # h = F.elu(item_embeds_4)
        # item_embeds_5 = self.dropout5(F.elu(self.GConv5(g, item_embeds_4, edge_weight=g.edata['w'])))
        # item_embeds_5 = torch.flatten(self.GConv5(g, h), 1)
        item_embeds = torch.cat([item_embeds_1, item_embeds_2, item_embeds_3], 1)
        # item_embeds = self.i_mix(F.elu(torch.cat([item_embeds_1, item_embeds_2], 1))) # item_embeds_4 # item_embeds_2, item_embeds_3
        # , item_embeds_4, item_embeds_5

        # item_embeds = self.fc(
        #     torch.cat([item_embeds_1, item_embeds_2, item_embeds_3], 1))  # item_embeds_2, item_embeds_3


        # mean vector aggregate .. [0 1 0 1 1] 아래는 mean aggregator 코드
        # item_cnt = torch.sum(queries, dim=1).reshape(1, -1)
        # # tmp = item_embeds[queries]
        #
        # query_embeds = torch.matmul(queries, item_embeds)
        # query_embeds = torch.div(query_embeds.T, item_cnt).T

        # 쿼리 = [10, 5, 998, 0, 0], length = 3
        query_embeds = item_embeds[query_items] # batch, 4, 256

        input_batch = pack_padded_sequence(query_embeds, lengths.tolist(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.mix_lstm(input_batch)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        query_embeds = output[range(len(output)), lengths - 1, :]
        # query_embeds = self.lstm_mix(F.elu(output[range(len(output)), lengths - 1, :])) # batch, emb * 2

        pos_embed = item_embeds[pos_items, :]
        neg_embed = item_embeds[neg_items, :]  # item_embeds[neg_items[:, 0], :]

        # batch size별로 negative가 여러개...
        # tmp = item_embeds[neg_items, :]

        # import pdb; pdb.set_trace()

        # query, pos, neg, ranked pos embedding, ranked neg emebedding

        # query_item x pos_item
        # q x p = > i == j postive, batch_size - 1 만큼은 negative로 한다.
        # query (item군으로 들어오는 것이 있는가?)

        # all score
        # tmp = torch.flatten(F.relu(self.i_mix2(item_embeds))).expand(query_embeds.shape[0], -1)
        # tmp = tmp.expand(query_embeds.shape[0], -1)
        # logit = self.q_fc(F.elu(query_embeds)) @ self.q_fc(F.elu(item_embeds)).T # query에 대한 각 아이템의 score
        logit = query_embeds @ item_embeds.T  # query에 대한 각 아이템의 score
        # (query_embeds, score) # query embedding, 각 score를 주는 거
        # logit = self.ranker_out(F.elu(self.ranker(F.elu(torch.cat([query_embeds, torch.flatten(F.relu(self.i_mix2(item_embeds))).expand(query_embeds.shape[0], -1)], 1)))))

        # item_embeds # 42000 x dim

        return query_embeds, pos_embed, neg_embed, logit

    def get_bpr_loss(self, query_item, pos_items, neg_items, lmda=1e-5):  # 1e-5)
        pos_scores = (query_item * pos_items).sum(1)
        neg_scores = (query_item * neg_items[:, 0]).sum(1)


        # negative sampling loss
        # tmp = query_item.view(-1, 1, query_item.shape[1]).repeat(1, neg_items.shape[1], 1)
        # .repeat(1, neg_items.shape[1], 1, )
        # neg_sample_scores = (
        #             query_item.view(-1, 1, query_item.shape[1]).repeat(1, neg_items.shape[1], 1) * neg_items).sum(
        #     2)  # 1024 * 128, 1024 * 4 * 128
        # neg_sample_softmax_loss = -1 * torch.log_softmax(torch.cat([pos_scores.view(-1, 1), neg_sample_scores], dim=1), dim=1)[0].mean()
        # pos_scores = self.cos(query_item, pos_items)
        # neg_scores = self.cos(query_item, neg_items)

        # query_item * pos_items

        # max(0, sqrt(((query_item - pos_items) ** 2).sum()) - sqrt(((pos_items - neg_items) ** 2).sum()) + e)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()  # log2 @ log -2
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(query_item) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = lmda * regularizer / query_item.shape[0]
        # mf_loss + emb_loss
        return mf_loss + emb_loss, mf_loss, emb_loss, 0 # neg_sample_softmax_loss
