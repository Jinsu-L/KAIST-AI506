import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.GAT import GAT


class Task2Net(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.embedding = nn.Embedding(len(g.nodes()), embedding_dim=64)
        self.GAT = GAT(g, in_dim=64, hidden_dim=8, out_dim=16, num_heads=3)

    def forward(self, queries, pos_items, neg_items):
        item_embeds = self.GAT(self.embedding.weight)

        # mean vector aggregate ..
        item_cnt = torch.sum(queries, dim=1).reshape(1, -1)
        query_embeds = torch.matmul(queries, item_embeds)
        query_embeds = torch.div(query_embeds.T, item_cnt).T

        pos_embed = item_embeds[pos_items, :]
        neg_embed = item_embeds[neg_items, :]

        return query_embeds, pos_embed, neg_embed

    def get_bpr_loss(self, query_item, pos_items, neg_items, lmda=1e-5):
        pos_scores = (query_item * pos_items).sum(1)
        neg_scores = (query_item * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(query_item) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = lmda * regularizer / query_item.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss
