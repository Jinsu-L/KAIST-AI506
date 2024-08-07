"""
    Task 1 : Outfit Generation dataset 생성 코드
"""

import argparse
import os
import logging
import math
import copy
import numpy as np
import random

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

from tqdm import tqdm
from collections import defaultdict


def main(args):
    logging.info("Experiment args : " + str(args))

    # read dataset

    user_map = dict()
    itemset_map = dict()

    train = defaultdict(list)
    item_cnt = defaultdict(int)
    
    # read train
    with open(os.path.join(args.input_dir, "user_item.csv")) as f: # "user_itemset_training.csv"
        for line in tqdm(f):
            user_id, itemset_id = line.strip().split(",")
            item_cnt[itemset_id] += 1
            if user_id not in user_map.keys():
                user_map[user_id] = user_id # user_idx
                user_key = user_id
                # user_idx += 1
            else:
                user_key = user_map[user_id]

            if itemset_id not in itemset_map.keys():
                itemset_map[itemset_id] = itemset_id
                itemset_key = itemset_id # itemset_idx
                # itemset_idx += 1
            else:
                itemset_key = itemset_map[itemset_id]

            train[user_key].append(itemset_key)

    # valid => test.txt
    """
        user_itemset test 생성시 주석 해제
    """
    # test = defaultdict(list)
    # with open(os.path.join(args.input_dir, "user_itemset_valid_query.csv")) as f,  open(os.path.join(args.input_dir, "user_itemset_valid_answer.csv")) as f2:
    #     for line1, line2 in zip(f, f2):
    #         user_id, itemset_id = line1.strip().split(",")
    #         label = line2.strip()
    # 
    #         if label == "1":
    #             test[user_map[user_id]].append(itemset_map[itemset_id])

    # write
    with open(os.path.join(args.output_dir, "user_list.txt"), "w") as w:
        # write head
        w.write("org_id remap_id\n")

        for user_id, user_idx in user_map.items():
            w.write(" ".join([user_id, str(user_idx)]) + "\n")

    with open(os.path.join(args.output_dir, "item_list.txt"), "w") as w:
        # write head
        w.write("org_id remap_id\n")

        for itemset_id, itemset_idx in itemset_map.items():
            w.write(" ".join([itemset_id, str(itemset_idx)]) + "\n")

    # train, test split
    with open(os.path.join(args.output_dir, "train.txt"), "w") as train_w, open(os.path.join(args.output_dir, "test.txt"), "w") as test_w:
        for user_id, itemset_id_list in train.items():
            train_buf = []
            test_buf = []
            for itemset_id in itemset_id_list:
                prob = random.random()
                if prob > args.test_rate and item_cnt[itemset_id] > 1:
                    train_buf.append(itemset_id)
                else:
                    test_buf.append(itemset_id)

            if train_buf:
                train_w.write(str(user_id) + " " + " ".join(map(str, train_buf)) + "\n")
            if test_buf:
                test_w.write(str(user_id) + " " + " ".join(map(str, test_buf)) + "\n")

    # user-itemset 생성시 사용
    # with open(os.path.join(args.output_dir, "train.txt"), "w") as w:
    #     for user_id, itemset_id_list in train.items():
    #         w.write(str(user_id) + " " + " ".join(map(str, itemset_id_list)) + "\n")
    #
    # with open(os.path.join(args.output_dir, "test.txt"), "w") as w:
    #     for user_id, itemset_id_list in train.items():
    #         w.write(str(user_id) + " " + " ".join(map(str, itemset_id_list)) + "\n")

    logging.info("end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./Dataset")
    parser.add_argument("--output_dir", type=str, default="./NGCF-DGL/Data/item-recommendation")
    parser.add_argument("--test_rate", type=float, default=0.2)

    args = parser.parse_args()

    main(args)
