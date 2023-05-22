from collections import defaultdict
from tqdm import tqdm

path = "./Dataset"

# relation info
train_file = path + '/itemset_item_training.csv'

# 전체 train 데이터를 읽어서, item-item graph를 만든다.
itemset_items = defaultdict(list)
item_itemsets = defaultdict(list)
exist_items = set()
with open(train_file) as f:
    for line in f:
        itemset_id, item_id = line.strip().split(",")
        itemset_items[int(itemset_id)].append(int(item_id))
        item_itemsets[int(item_id)].append(int(itemset_id))
        exist_items.add(int(item_id))

item_value = dict()
for i in tqdm(range(0, max(exist_items) + 1)):
    for j in range(i, max(exist_items) + 1):
        if i == j:
            continue

        i_set = set(item_itemsets[i])
        j_set = set(item_itemsets[j])

        intersection = len(i_set.intersection(j_set))

        if intersection:
            if intersection:
                union = len(i_set.union(j_set))
                item_value[(i, j)] = intersection / union

with open(path + "/item_to_item_its_jac.csv", "w") as w:
    for (i, j), score in item_value.items():
        w.write("%d,%d,%f\n" %(i, j, score))