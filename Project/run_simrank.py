"""
    Task 1 : Outfit Generation Baseline 실행 코드
    Simrank
"""

import argparse
import os
import logging
import math
import copy
import numpy as np
import networkx as nx
import scipy.sparse

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

from collections import defaultdict

from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score


def _load_preference_data(path):
    user_preference = defaultdict(list)
    max_user = -1
    max_iid = -1

    with open(path) as f:
        for line in f:
            user, iid = list(map(int, line.strip().split(",")))
            user_preference[user].append([iid, 1])  # item_id, value
            max_user = max(max_user, user)
            max_iid = max(max_iid, iid)

    return user_preference, max_user, max_iid


def _make_sparse_matrix(user_preference, max_user_id, max_itemset_id):
    row = []
    col = []
    value = []

    for user, item_list in tqdm(user_preference.items()):
        for iid, score in item_list:
            row.append(user)
            col.append(iid)
            value.append(score)

    return coo_matrix((value, (row, col)), shape=(max_user_id + 1, max_itemset_id + 1))

def _make_networkx_graph(user_preference):
    G = nx.Graph()

    # memory 때문에 idx로 바꿔서 쓰기.
    idx_to_user = dict()
    user_to_idx = dict()
    idx_to_item = dict()
    item_to_idx = dict()

    edges = []
    idx = 0
    for user, item_list in tqdm(user_preference.items()):
        # user_id = "u_" + str(user)

        if user in user_to_idx.keys():
            user_idx = user_to_idx[user]
        else:
            idx_to_user[idx] = user
            user_to_idx[user] = idx
            user_idx = idx
            idx += 1

        for iid, score in item_list:
            # item_id = "I_" + str(iid)

            if iid in item_to_idx.keys():
                item_id = item_to_idx[iid]
            else:
                idx_to_item[idx] = iid
                item_to_idx[iid] = idx
                item_id = idx
                idx += 1

            edges.append((user_idx, item_id, score))
    G.add_weighted_edges_from(edges)

    return G # , idx_to_user, user_to_idx, idx_to_item, item_to_idx

def _add_negative_sample(user_preference, negative_sample_rate, max_itemset_id):
    neg_items = defaultdict(set)
    for user, item_list in user_preference.items():
        pos_items = set([iid for [iid, score] in item_list])
        neg_item_size = math.ceil(len(pos_items) * negative_sample_rate)

        while True:
            if len(neg_items[user]) == neg_item_size: break
            neg_item_id = np.random.randint(low=0, high=max_itemset_id, size=1)[0]

            if (neg_item_id not in pos_items) and (neg_item_id not in neg_items[user]):
                neg_items[user].add(neg_item_id)

    result = dict()

    for user, neg_item_id_set in neg_items.items():
        result[user] = copy.copy(user_preference[user]) + [[neg_item_id, -1] for neg_item_id in neg_item_id_set]

    return result

def _load_predset(query_path, max_user_id, max_itemset_id, answer_path=None):
    buf = []

    row = []
    col = []
    value = []

    # valid_graph, labels

    with open(query_path) as f:
        for line in tqdm(f):
            user, iid = list(map(int, line.strip().split(",")))
            buf.append([user, iid])

            row.append(user)
            col.append(iid)
            value.append(1)

    graph = coo_matrix((value, (row, col)), shape=(max_user_id + 1, max_itemset_id + 1))

    if answer_path is not None:
        new_buf = []
        with open(answer_path) as f:
            for v, line in zip(buf, f):
                new_buf.append(v + [bool(int(line.strip()))])

        return graph, new_buf
    else:
        return graph, buf

def load_dataset(args, is_nx=False):
    # load train
    logging.info("Load train Dataset")
    train, max_user_id, max_itemset_id = _load_preference_data(
        os.path.join(args.input_dir, "user_itemset_training.csv"))
    # train = _add_negative_sample(train, args.negative_sample_rate, max_itemset_id)

    if is_nx:
        train_graph = _make_networkx_graph(train) # networkx
    else:
        train_graph = _make_sparse_matrix(train, max_user_id, max_itemset_id)

    logging.info("Load valid Dataset")
    valid_graph, valid = _load_predset(os.path.join(args.input_dir, "user_itemset_valid_query.csv"), max_user_id,
                                       max_itemset_id,
                                       os.path.join(args.input_dir,
                                                    "user_itemset_valid_answer.csv"))  # user, item, label

    logging.info("Load test Dataset")
    test_graph, test = _load_predset(os.path.join(args.input_dir, "user_itemset_test_query.csv"), max_user_id,
                                     max_itemset_id)

    return train_graph, valid_graph, test_graph, valid, test

import itertools

def simrank(G, r=0.8, max_iter=100, eps=1e-4):
    # import pdb;pdb.set_trace()
    nodes = G.nodes()
    # nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes), dtype=np.half)
    sim_prev = np.copy(sim)
    sim = sim.dot(sim_prev).dot(sim.T)


    for i in range(max_iter):
        # if np.allclose(sim, sim_prev, atol=eps):
        #     break
        sim_prev = np.copy(sim)
        for u, v in tqdm(itertools.product(nodes, nodes), total=len(nodes) * len(nodes)):
            if u is v:
                continue
            u_ns, v_ns = G[u], G[v]
            continue

            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0:
                # if a node has no predecessors then setting similarity to zero
                # sim[nodes_i[u]][nodes_i[v]] = 0
                sim[u][v] = 0
            else:
                s_uv = sum([sim_prev[u_n][v_n] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[u][v] = (r * s_uv) / (len(u_ns) * len(v_ns))

    return sim

def main(args):
    logging.info("Experiment args : " + str(args))

    accuracy_list = []
    precision_list = []
    recall_list = []

    # from scipy.sparse import identity
    # sim = identity(82000)
    # sim = np.identity(82000)

    # import pdb; pdb.set_trace()

    for i in range(args.num_experiments):
        logging.info("[START] Experiment " + str(i+1) + " ")

        # train, valid, test = load_dataset(args)
        train_graph, valid_graph, test_graph, valid, test = load_dataset(args, is_nx=True)

        logging.info("Create Model")
        # model = LightFM(loss=args.loss, no_components=args.no_components)
        # model = nx.simrank_similarity(train_graph, max_iterations=10, source=0) # 계산하는데, 메모리가 엄청 들어가는데..
        sim = simrank(train_graph)
        logging.info("end Model")
        import pdb; pdb.set_trace()

        logging.info("Train Start")
        # model.fit(train_graph, epochs=args.epochs, num_threads=args.num_threads)
        logging.info("Train End")

        # Evaluation
        logging.info("Evaluation : Ranking for Valid")
        # rank_matrix = model.predict_rank(valid_graph)
        # dok_rank_matrix = rank_matrix.todok()

        logging.info("Evaluation : Pred for Valid")
        # pred_list = []
        # label_list = []
        # for user_id, itemset_id, label in tqdm(valid):
        #     pred_rank = dok_rank_matrix[user_id, itemset_id]
        #     pred = pred_rank <= args.rank_threashold
        #     pred_list.append(pred)
        #     label_list.append(label)

        # import pdb; pdb.set_trace()

        # report = classification_report(y_true=label_list, y_pred=pred_list, target_names=["False","True"], digits=4)
        # accuracy = accuracy_score(label_list, pred_list)
        # precision = precision_score(label_list, pred_list, pos_label=1)
        # recall = recall_score(label_list, pred_list, pos_label=1)

        # print("accuracy : ", accuracy)
        # print("precision : ", precision)
        # print("recall : ", recall)
        # print("f1-score : ", 2 * precision * recall / (precision + recall))
        #
        # accuracy_list.append(accuracy)
        # precision_list.append(precision)
        # recall_list.append(recall)


    logging.info("Experiment Statistics")
    for i, (accuracy, precision, recall) in enumerate(zip(accuracy_list, precision_list, recall_list)):
        print("Experiment ", i+1, " ", " Acc : ", accuracy, " Pre : ", precision, " Rec : ", recall)

    print()
    print("max: ", "Acc : ", max(accuracy_list), " Pre : ", max(precision_list), " Rec : ", max(recall_list))
    print("min: ", "Acc : ", min(accuracy_list), " Pre : ", min(precision_list), " Rec : ", min(recall_list))
    print("avg: ", "Acc : ", np.mean(accuracy_list), " Pre : ", np.mean(precision_list), " Rec : ", np.mean(recall_list))
    print("std: ", "Acc : ", np.std(accuracy_list), " Pre : ", np.std(precision_list), " Rec : ", np.std(recall_list))

    # generate test answer

    logging.info("End Experiments")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="bpr")
    parser.add_argument("--no_components", type=int, default=32)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rank_threashold", type=int, default=5000)
    parser.add_argument("--use_negative_sample", action="store_true")
    # todo: show top k 만들기

    parser.add_argument("--negative_sample_rate", type=float, default=9.0, help="유저의 interaction 대비 negative sample 비율")
    parser.add_argument("--num_experiments", type=int, default=1, help="테스트 평균과 분산 통계를 계산하기위한 실험 수")

    parser.add_argument("--input_dir", type=str, default="./Dataset")
    parser.add_argument("--output_dir", type=str, default="./Experiments")

    args = parser.parse_args()

    main(args)
