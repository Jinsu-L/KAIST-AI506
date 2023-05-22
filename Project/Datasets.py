import os
# import pytorch_lightning.pytorch as pl
import torch
import itertools
import copy
import random
import numpy as np
import dgl
from torch.utils.data import Dataset
from collections import defaultdict


class Task2Dataset(Dataset):
    def __init__(self, path):
        self.path = path

        # relation info
        train_file = path + '/itemset_item_training.csv'
        item_to_item_file = path + '/item_to_item_its_jac.csv'
        valid_file = path + '/itemset_item_valid_query.csv'
        valid_answer_file = path + '/itemset_item_valid_answer.csv'

        # get number of users and items
        self.rel_cnt = 0
        self.exist_items = set()
        self.max_item_id = -1

        item_item_src = []
        item_item_dst = []
        item_item_weights = []

        # graph relation
        with open(item_to_item_file) as f:
            for line in f:
                item_src, item_dst, score = line.strip().split(",")
                item_item_src.append(int(item_src))
                item_item_dst.append(int(item_dst))
                item_item_weights.append(float(score))
                self.rel_cnt += 1

        # training dataset
        self.train_itemset_items = defaultdict(list)
        self.its2idx = dict()

        with open(train_file) as f:
            for line in f:
                itemset_id, item_id = line.strip().split(",")

                if int(itemset_id) in self.its2idx.keys():
                    its_id = self.its2idx[int(itemset_id)]
                else:
                    self.its2idx[int(itemset_id)] = len(self.its2idx.keys())
                    its_id = self.its2idx[int(itemset_id)]

                self.train_itemset_items[its_id].append(int(item_id))
                self.exist_items.add(int(item_id))
                self.max_item_id = max(self.max_item_id, int(item_id))

        self.n_items = len(self.exist_items)

        # find keys : pos
        self.train_pos_keys = defaultdict(list)
        for its_id, item_list in self.train_itemset_items.items():
            item_set = set(item_list)
            for e in itertools.combinations(item_list, len(item_list) - 1):  # 각 조합
                key = "\t".join(map(str, sorted(e)))
                self.train_pos_keys[key].append(list(item_set - set(e))[0])

        # valid dataset
        self.valid_itemset_items = defaultdict(list)
        self.valid_itemset_label = dict()
        with open(valid_file) as f:
            for line in f:
                itemset_id, item_id = line.strip().split(",")
                self.valid_itemset_items[int(itemset_id)].append(int(item_id))

        with open(valid_answer_file) as f:
            for line in f:
                itemset_id, item_id = line.strip().split(",")
                self.valid_itemset_label[int(itemset_id)] = int(item_id)

        # construct graph from the train data and add self-loops
        for i in range(self.max_item_id + 1):
            item_item_src.append(i)
            item_item_dst.append(i)
            item_item_weights.append(1.0)  # jaccard max

        self.g = dgl.graph((torch.tensor(item_item_src), torch.tensor(item_item_dst)))
        self.g.edata['w'] = torch.tensor(item_item_weights)


    def __len__(self):
        return len(self.train_itemset_items)

    def __getitem__(self, itemset_id):
        # itemsets, query_items, pos_item(label), neg_item(random)
        target_items = self.train_itemset_items[itemset_id]

        # negative item은 query item or pos item이 아닌 것?
        pos = random.choice(target_items)
        query_items = copy.copy(target_items)
        query_items.remove(pos)
        key = "\t".join(map(str, sorted(query_items)))
        pos_candidate = self.train_pos_keys[key]

        # 0 or 1 vector [0, 1 .... , 1... 0] query item에만 1로 해서 itemembedding이랑 @ 해서 뽑을 수 있게?
        query = np.zeros(self.n_items, dtype=np.float32)
        query[query_items] = 1

        while True:
            neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_id not in target_items and neg_id not in pos_candidate:
                neg = neg_id
                break
        return itemset_id, query, pos, neg
