import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from tqdm import tqdm, trange
from Datasets import Task1Dataset
from Models.Task1 import Task1Net
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import copy

if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'

###########################################################

save_file = "ours_all_81.3318.tar"

###########################################################

# def calc_valid()
def get_accuracy(y_true, y_prob):
    y_true = y_true == 1.0
    y_prob = y_prob >= 0.5

    acc = torch.true_divide((y_prob == y_true).sum(dim=0), y_true.size(0)).item()

    y_true = y_true.detach().cpu().numpy()
    y_prob = y_prob.detach().cpu().numpy()

    precision = precision_score(y_true, y_prob)
    recall = recall_score(y_true, y_prob)
    f1 = f1_score(y_true, y_prob)

    return acc, precision, recall, f1

dataset = Task1Dataset("Dataset/")
train_dataloader = DataLoader(dataset, batch_size=4096*2, shuffle=True)

valid_dataset = Task1Dataset("Dataset/", is_train=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)

u_e = torch.tensor(dataset.user_embedding).to(device)
i_e = torch.tensor(dataset.item_embedding).to(device)
s_e = torch.tensor(dataset.style_embedding).to(device)

model = Task1Net(dataset.max_itemset_id+1, dataset.max_user_id+1, u_e, i_e, s_e).to(device)

state_dict = torch.load(save_file)
model.load_state_dict(state_dict['model'])

model.eval()
with torch.no_grad():
    pred_list = []
    true_list = []
    for user_ids, itemset_ids, labels, item_list, length_list, pop in valid_dataloader:
#         print(user_ids[0], itemset_ids[0])
        outputs = model(user_ids.to(device), itemset_ids.to(device), item_list.to(device), length_list.to(device), pop.to(device))
        pred_list.append(outputs)
        true_list.append(labels.to(device))

    pred_list = torch.cat(pred_list)
    true_list = torch.cat(true_list)
    accuracy, precision, recall, f1 = get_accuracy(true_list, pred_list)

print("valid score")
print(*list(map(lambda e: round(e * 100,4),[accuracy, precision, recall, f1])))

# generate test pred
test_dataset = Task1Dataset("Dataset/", is_train=False, gen_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

pred_list = []
with torch.no_grad():
    for user_ids, itemset_ids, item_list, length_list, pop in test_dataloader:
        outputs = model(user_ids.to(device), itemset_ids.to(device), item_list.to(device), length_list.to(device),
                        pop.to(device))
        pred_list.append(outputs)
pred_list = torch.cat(pred_list)

with open("user_itemset_test_prediction.csv", "w") as w:
    for l in pred_list.detach().cpu().numpy() > 0.5:
        w.write(str(1 if l else 0) + "\n")
