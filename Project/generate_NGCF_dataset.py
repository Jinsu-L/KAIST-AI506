"""
    Task 1 : Outfit Generation dataset 생성 코드
"""

import argparse
import os
import logging
import math
import copy
import numpy as np

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
    user_idx = 0
    itemset_idx = 0

    train = defaultdict(list)

    # read train
    with open(os.path.join(args.input_dir, "user_itemset_training.csv")) as f:
        for line in tqdm(f):
            user_id, itemset_id = line.strip().split(",")
            if user_id not in user_map.keys():
                user_map[user_id] = user_idx
                user_key = user_idx
                user_idx += 1
            else:
                user_key = user_map[user_id]

            if itemset_id not in itemset_map.keys():
                itemset_map[itemset_id] = itemset_idx
                itemset_key = itemset_idx
                itemset_idx += 1
            else:
                itemset_key = itemset_map[itemset_id]

            train[user_key].append(itemset_key)

    # valid => test.txt
    test = defaultdict(list)
    with open(os.path.join(args.input_dir, "user_itemset_valid_query.csv")) as f,  open(os.path.join(args.input_dir, "user_itemset_valid_answer.csv")) as f2:
        for line1, line2 in zip(f, f2):
            user_id, itemset_id = line1.strip().split(",")
            label = line2.strip()

            if label == "1":
                test[user_map[user_id]].append(itemset_map[itemset_id])

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

    with open(os.path.join(args.output_dir, "train.txt"), "w") as w:
        for user_id, itemset_id_list in train.items():
            w.write(str(user_id) + " " + " ".join(map(str, itemset_id_list)) + "\n")

    with open(os.path.join(args.output_dir, "test.txt"), "w") as w:
        for user_id, itemset_id_list in test.items():
            w.write(str(user_id) + " " + " ".join(map(str, itemset_id_list)) + "\n")

    logging.info("end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./Dataset")
    parser.add_argument("--output_dir", type=str, default="./NGCF-DGL/Data/itemset-recommendation")

    args = parser.parse_args()

    main(args)
