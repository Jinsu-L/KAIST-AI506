import os
# import pytorch_lightning.pytorch as pl
import torch
import itertools
import copy
import random
import numpy as np
import dgl
from torch.utils.data import Dataset
from torch import nn
from collections import defaultdict

class Task1Dataset(Dataset):
    def __init__(self, path, is_train=True):
        self.path = path
        self.is_train = is_train

        train_file = path + "/user_itemset_training.csv" # 여기 있는 건 전부 True
        valid_query_file = path + "/user_itemset_valid_query.csv"
        valid_answer_file = path + "/user_itemset_valid_answer.csv"

        user_embedding_file = path + "/user_embedding_ngcf16_16_16_16.npy" # ngcf에서 만든거.. USER, ITEM에 대한 임베딩
        item_embedding_file = path + "/item_embedding_ngcf16_16_16_16.npy"
        style_embedding_file = path + "/CASG_GNN_ITEM_OUT.npy" # CASG에서 만든 item에 대한 임베딩.
        itemset_item_file = path + "/itemset_item_training.csv"


        self.max_user_id = 0
        self.max_itemset_id = 0

        self.train = []
        self.true_sample = []
        self.true_user_itemset = defaultdict(list)
        with open(train_file) as f:
            for line in f:
                user_id, itemset_id = list(map(int, line.strip().split(",")))
                self.max_user_id = max(self.max_user_id, user_id)
                self.max_itemset_id = max(self.max_itemset_id, itemset_id)

                self.true_sample.append([user_id, itemset_id, 1])
                self.true_user_itemset[user_id].append(itemset_id)

        # negative sample generate
        # 어떤 유저가 itemset을 사용한 거에 대하여... 음.. 어케 하지?
        # random 한 유저, itemset pair를 잡아서.. 0으로 라벨하는 거지.
        self.regenerated_neg_sample()

        self.valid_user_ids = []
        self.valid_itemset_ids = []
        self.valid_labels = []
        self.valid = []
        with open(valid_query_file) as qf, open(valid_answer_file) as af:
            for q_line , a_line in zip(qf, af):
                user_id, itemset_id = list(map(int, q_line.strip().split(",")))
                answer = int(a_line.strip())
                self.valid.append([user_id, itemset_id, answer])
                self.valid_user_ids.append(user_id)
                self.valid_itemset_ids.append(itemset_id)
                self.valid_labels.append(answer)

        self.valid_user_ids = torch.tensor(self.valid_user_ids)
        self.valid_itemset_ids = torch.tensor(self.valid_itemset_ids)
        self.valid_labels = torch.tensor(self.valid_labels)

        # NGCF embedding
        self.item_embedding = np.load(item_embedding_file) # (n_items, 64)
        self.user_embedding = np.load(user_embedding_file) # (n_users, 64)
        self.style_embedding = np.load(style_embedding_file) # (n_itemsets, 64)

        # itemset - item mapping file
        self.itemset_d = defaultdict(list)
        with open(itemset_item_file) as f:
            for line in f:
                itemset_id, item_id = line.strip().split(",")
                self.itemset_d[int(itemset_id)].append(int(item_id))


    def regenerated_neg_sample(self):
        self.false_sample = []

        # 유저별로 샘플링?? 특정 유저에서는 sample이 부족할 수도? 근데 negative가 많아지면 그러니..
        # 각 유저별로 True 개수 맞추어서 negative sample을 생성.

        for user_id, true_items in self.true_user_itemset.items():
            neg_sample = []
            neg_size = len(true_items)
            while True:
                if len(neg_sample) == neg_size:
                    break

                neg_id = np.random.randint(low=0, high=self.max_itemset_id+1, size=1)[0]
                if (neg_id not in true_items) and (neg_id not in neg_sample):
                    neg_sample.append(neg_id)

            for sample in neg_sample:
                self.false_sample.append([user_id, sample, 0])

        self.train = self.true_sample + self.false_sample

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def __getitem__(self, sample_id):
        if self.is_train:
            user_id, itemset_id, label = self.train[sample_id]
            # return user_id, itemset_id, torch.tensor([label], dtype=torch.float32)
        else:
            user_id, itemset_id, label = self.valid[sample_id]

        # itemset to item
        item_list = np.array(self.itemset_d[itemset_id]) # (1,3,9,55)
        length = len(item_list)
        item_list = list(item_list[np.random.permutation(len(item_list))])
        item_list += [0 for i in range(5 - len(item_list))] # max item = 5

        # torch.tensor(list(np.array(query_items)[np.random.permutation(len(query_items))]) + [0 for i in range(
        #     5 - len(query_items))]), len(query_items)

        return user_id, itemset_id, torch.tensor([label], dtype=torch.float32), torch.tensor(item_list), length



class Task2Dataset(Dataset):
    def __init__(self, path):
        self.path = path

        # '/item_to_item_its_jac.csv' : 기본용
        #'/item_by_item_its_jac_with_mst.csv' : mst 용

        # relation info
        train_file = path + '/itemset_item_training.csv'
        item_to_item_file = path + '/item_by_item_its_jac_with_mst.csv' # 'item_by_item_its_jac_with_mst.csv' # '/item_to_item_its_jac.csv'
        valid_file = path + '/itemset_item_valid_query.csv'
        valid_answer_file = path + '/itemset_item_valid_answer.csv'
        mst_file = path + 'mst.csv'
        feature_file = path + 'louvain_community_feature.csv'

        # get number of users and items
        self.rel_cnt = 0
        self.exist_items = set()
        self.max_item_id = -1

        self.item_item_src = []
        self.item_item_dst = []
        self.item_item_weights = []

        # graph relation
        with open(item_to_item_file) as f:
            for line in f:
                item_src, item_dst, score = line.strip().split(",")
                self.item_item_src.append(int(item_src))
                self.item_item_dst.append(int(item_dst))
                self.item_item_weights.append(float(score))
                self.rel_cnt += 1

        #read mst
        self.mst_relation = set()
        with open(mst_file) as f:
            for line in f:
                node_1, node_2 = line.strip().split(",") # str 타입
                self.mst_relation.add(node_1 + "\t" + node_2)

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
            self.item_item_src.append(i)
            self.item_item_dst.append(i)
            self.item_item_weights.append(1.0)  # jaccard max

        self.g = dgl.graph((torch.tensor(self.item_item_src), torch.tensor(self.item_item_dst)))
        self.g.edata['w'] = torch.tensor(self.item_item_weights)

        # read feature file
        # 각 node별로 depth 별로 파일 읽어서 처리
        # 291,0,0,69,0,None
        # 2043,0,0,69,0,None
        # feature가 없는 itemnode는 올 None 처리.
        # 각 feature 별로 최대 cluster 개수 sum + None 해서 onehot label으로 해서 feature
        # None인 경우는 one-hot feature에서 처리 X 즉 1번 케이스에서는 4개만 1

        self.max_feature_id = -1
        buf = dict()
        with open(feature_file) as f:
            for line in f:
                tkns = line.strip().split(",")
                item_id = int(tkns[0])
                feature_ids = []
                for f in tkns[1:]:
                    if f != 'None':
                        feature_ids.append(int(f))
                        self.max_feature_id = max(self.max_feature_id, int(f))

                buf[item_id] = feature_ids

        self.node_feature = torch.zeros(len(self.g.nodes()), self.max_feature_id + 1)

        # gen one-hot
        for item_id, feature_ids in buf.items():
            self.node_feature[item_id][feature_ids] = 1

        self.g.ndata['feature'] = self.node_feature


    def __len__(self):
        # return len(self.train_itemset_items)
        return len(self.train)

    def resample(self, train_rate=0.35):
        # 전체 데이터에서 drop_rate 만큼을 train itemset으로 쓴다.
        # 그리고 나머지 item에 대하여만, train_g를 구성한다.

        train_idx = set(np.random.choice(list(self.train_itemset_items.keys()), int(train_rate * len(self.train_itemset_items.keys())), replace=False))

        self.train = []

        new_item_itemset_d = defaultdict(set)

        for itemset_id, target_items in self.train_itemset_items.items():
            
            # train set인 경우만 학습에 사용
            if itemset_id in train_idx:
                pos = random.choice(target_items)
                query_items = copy.copy(target_items)
                query_items.remove(pos)
                key = "\t".join(map(str, sorted(query_items)))
                pos_candidate = self.train_pos_keys[key]

                query = np.zeros(self.n_items, dtype=np.float32)
                query[query_items] = 1

                neg = []
                while True:
                    if len(neg) == 99:
                        break
                    neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                    if neg_id not in target_items and neg_id not in pos_candidate:
                        neg.append(neg_id)
                
                # self.train[itemset_id] = [pos, torch.tensor(neg), query, query_items]
                self.train.append([pos, torch.tensor(neg), query, query_items])
            else:
                # train에 포함되지 않는 경우의 relation만 Graph에 반영.
                # item : itemsets
                for item_id in target_items:
                    new_item_itemset_d[item_id].add(itemset_id)

        # Calc Relation
        new_src = []
        new_dst = []
        new_weight = []
        for src, dst, weight in zip(self.item_item_src, self.item_item_dst, self.item_item_weights):

            if src == dst: # self-loop
                new_src.append(src)
                new_dst.append(dst)
                new_weight.append(weight)
                continue


            src_itemset = new_item_itemset_d[src] # 여기에 있는 거면, train sample 노드 라는 것.
            dst_itemset = new_item_itemset_d[dst]

            # MST 노드인 경우, intersection이 0이여도 relation을 만들어야함.
            # trainset에 뽑히지 않은 것 때문에, relation이 아예 0인 노드라도, 1개가 있는 것으로
            if (str(src) + "\t" + str(dst) in self.mst_relation) or (str(src) + "\t" + str(dst) in self.mst_relation):
                intersection = len(src_itemset.intersection(dst_itemset))
                union = len(src_itemset.union(dst_itemset))
                new_src.append(src)
                new_dst.append(dst)
                new_weight.append((intersection+1) / (union+1))
            else:
                intersection = len(src_itemset.intersection(dst_itemset))

                if intersection:
                    union = len(src_itemset.union(dst_itemset))
                    new_src.append(src)
                    new_dst.append(dst)
                    new_weight.append(intersection / union)

        self.train_g = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)))
        self.train_g.edata['w'] = torch.tensor(new_weight)

    # def resample(self, drop_rate=0.5):
    #     # 전체 train edge?에 대하여 랜덤하게 drop, 다 지우려고 하니까, 자기네들끼리 있는 케이스가 문제
    #     # 특정 에폭 or 기준 마다 호출시 G를 새로 생성, pos를 다시 생성.
    #     # self.g => init에서 만든 전체 relation graph
    #     # self.re_g => resample로 만들어진 positive에 대한 graph
    #     # self.train => query랑 positive가 있도록하고, neg도 미리 만들어두고.. 아니면 neg는 그때 그때 해도... 동일할듯??
    #
    #     self.train = dict()
    #
    #     pos_related_edges = []
    #
    #     # gen case
    #     for itemset_id, target_items in self.train_itemset_items.items():
    #         pos = random.choice(target_items)
    #         query_items = copy.copy(target_items)
    #         query_items.remove(pos)
    #         key = "\t".join(map(str, sorted(query_items)))
    #         pos_candidate = self.train_pos_keys[key]
    #
    #         query = np.zeros(self.n_items, dtype=np.float32)
    #         query[query_items] = 1
    #
    #         neg = []
    #         while True:
    #             if len(neg) == 99:
    #                 break
    #             neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
    #             if neg_id not in target_items and neg_id not in pos_candidate:
    #                 neg.append(neg_id)
    #
    #         for q_item in query_items:
    #             pos_related_edges.append(tuple(sorted([pos, q_item])))
    #
    #         # itemset_id, pos, neg, query
    #         self.train[itemset_id] = [pos, torch.tensor(neg), query, query_items]
    #
    #     # remove select
    #     # remove_idx = np.random.choice(list(self.train.keys()), int(drop_rate * len(self.train.keys())), replace=False)
    #     # remove_key = set()
    #     # for idx in remove_idx:
    #     #     [pos, neg, query, query_items] = self.train[idx]
    #     #     for query_idx in query_items:
    #     #         remove_key.add(tuple(sorted([query_idx, pos])))
    #
    #     remove_idx = np.random.choice(list(range(len(pos_related_edges))), int(drop_rate * len(pos_related_edges)), replace=False)
    #     remove_key = set()
    #     for idx in remove_idx:
    #         remove_key.add(pos_related_edges[idx])
    #
    #     new_src = []
    #     new_dst = []
    #     new_weight = []
    #     for src, dst, weight in zip(self.item_item_src, self.item_item_dst, self.item_item_weights):
    #         if tuple(sorted([src, dst])) in remove_key:
    #             continue
    #         else:
    #             new_src.append(src)
    #             new_dst.append(dst)
    #             new_weight.append(weight)
    #
    #
    #     self.train_g = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)))
    #     self.train_g.edata['w'] = torch.tensor(new_weight)

    def __getitem__(self, itemset_id):
        # itemsets, query_items, pos_item(label), neg_item(random)
        # target_items = self.train_itemset_items[itemset_id]
        #
        # # negative item은 query item or pos item이 아닌 것?
        # pos = random.choice(target_items)
        # query_items = copy.copy(target_items)
        # query_items.remove(pos)
        # key = "\t".join(map(str, sorted(query_items)))
        # pos_candidate = self.train_pos_keys[key]
        #
        # # 0 or 1 vector [0, 1 .... , 1... 0] query item에만 1로 해서 itemembedding이랑 @ 해서 뽑을 수 있게?
        # query = np.zeros(self.n_items, dtype=np.float32)
        # query[query_items] = 1
        #
        # while True:
        #     neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
        #     if neg_id not in target_items and neg_id not in pos_candidate:
        #         neg = neg_id
        #         break
        #
        # return itemset_id, query, pos, neg

        [pos, neg, query, query_items] = self.train[itemset_id]

        return itemset_id, query, pos, neg, torch.tensor(list(np.array(query_items)[np.random.permutation(len(query_items))]) + [0 for i in range(5 - len(query_items))]), len(query_items)
