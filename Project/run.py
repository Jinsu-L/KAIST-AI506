import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from tqdm import tqdm
from Datasets import Task2Dataset
from Models.Task2 import Task2Net

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

dataset = Task2Dataset("Dataset/")

train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = Task2Net(dataset.g.to(device)).to(device)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(30)):
    loss, mf_loss, emb_loss = 0., 0., 0.
    for itemset_id, query, pos, neg in train_dataloader:
        query_embeds, pos_embeds, neg_embeds = model(query.to(device), pos.to(device), neg.to(device))

        total_loss, batch_mf_loss, batch_emb_loss = model.get_bpr_loss(query_embeds, pos_embeds, neg_embeds)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss

    # if (epoch + 1) % 5 != 0:
    perf_str = '\nEpoch %d : train==[%.5f=%.5f + %.5f]' % (epoch, loss, mf_loss, emb_loss)
    print(perf_str)

"""
valid에 대하여는 각자 아이템셋에 대하여, query vector 만들어서 top 100개를 만들었을때

"""