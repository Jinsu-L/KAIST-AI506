import torch
import numpy as np
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

###########################################################
batch_size = 65536
lr = 1e-3
epoch = 3000
scheduler_step = 300
scheduler_decay = 0.9

resample_rate = 0.35 # dataset에서 label로 쓰일 비율
resample_epoch = 10

save_folder = "./GCN_3L_CASG"

###########################################################

dataset = Task2Dataset("Dataset/")
dataset.resample(resample_rate)
g = dataset.g.to(device)
train_g = dataset.train_g.to(device)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = Task2Net(dataset.g.to(device)).to(device)

print(model)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_decay)

ce_loss = nn.CrossEntropyLoss()
pbar = tqdm(range(epoch))
for epoch in pbar:

    loss, mf_loss, emb_loss, cross_entopy_loss = 0., 0., 0., 0.

    # dataset re sample
    if epoch % resample_epoch == 0:
        dataset.resample(resample_rate)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_g = dataset.train_g.to(device)

    for itemset_id, query, pos, neg, query_items, lengths in train_dataloader:
        # label = nn.functional.one_hot(pos, num_classes=dataset.max_item_id + 1)

        # query_embeds : query에 대하여 LSTM 통과한 embedding (batch_size, dim)
        # pos_embeds : (batch_Size, dim)
        # neg_embeds : (batch_Size, dim)
        # logit : 모든 query by item (batch_size, n_items)
        query_embeds, pos_embeds, neg_embeds, logit = model(train_g, query.to(device), pos.to(device), neg.to(device),
                                                            query_items.to(device), lengths.to(device))


        # total_loss : batch_mf_loss (bpr) + batch_emb_loss (reg term)
        total_loss, batch_mf_loss, batch_emb_loss, _ = model.get_bpr_loss(query_embeds, pos_embeds,
                                                                       neg_embeds)
        # cross_entropy loss
        ce_batch_loss = ce_loss(logit, pos.to(device))

        optimizer.zero_grad()
        # total_loss.backward() # bpr loss
        # (total_loss + 0.1 * ce_batch_loss).backward()
        ce_batch_loss.backward() # CE loss
        optimizer.step()

        loss += total_loss + 0.1 * ce_batch_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss
        cross_entopy_loss +=ce_batch_loss

    if (epoch + 1) % 5 != 0:
        perf_str = '\nEpoch %d : train==[%.5f=%.5f + %.5f,CE=%.5f]' % (epoch, loss, mf_loss, emb_loss, cross_entopy_loss)
        # print(perf_str)
        pbar.set_postfix({"" : perf_str})

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            labels = []
            batched_query = []
            query_items = []
            lengths = []
            for itemset_id, querys in dataset.valid_itemset_items.items():
                query = np.zeros(dataset.n_items, dtype=np.float32)
                label = dataset.valid_itemset_label[itemset_id]
                query[querys] = 1

                batched_query.append(query)
                labels.append(label)
                query_items.append(querys + [0 for _ in range(5 - len(querys))])
                lengths.append(len(querys))

            # import pdb; pdb.set_trace()
            query_embeds, pos_embeds, _, logit = model(g, torch.tensor(batched_query).to(device), torch.tensor(labels).to(device),
                                                       neg.to(device), torch.tensor(query_items).to(device),
                                                       torch.tensor(lengths).to(device))
            valid_loss = ce_loss(logit, torch.tensor(labels).to(device))

            scores = torch.topk(logit, 100).indices.detach().cpu().numpy()
            acc = sum([label in top100 for top100, label in zip(scores, labels)]) / len(scores)

            scores = torch.topk(logit, 500).indices.detach().cpu().numpy()
            top500acc = sum([label in top100 for top100, label in zip(scores, labels)]) / len(scores)
            print('\nEpoch %d : lr=%.10f top100_acc=%.5f top500_acc=%.5f' % (epoch, optimizer.param_groups[0]['lr'], acc, top500acc))

            # print("lr: ", optimizer.param_groups[0]['lr'])
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_folder + "/model_%d_%.5f_%.5f.tar" % (epoch, acc, top500acc))
        model.train()
    scheduler.step()

