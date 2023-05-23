from collections import defaultdict
from tqdm import tqdm

import pandas as pd

path = "./Dataset"
total_number_user: int = 53897
total_number_itemset: int = 27694
total_number_item: int = 42563

jaccard_similarity = lambda s1, s2: len(s1 & s2) / len(s1 | s2)

# relation info
train_file = path + '/user_item.csv'
user_item = pd.read_csv(train_file, delimiter=',', names=['user_id', 'item_id'])

# 전체 user-item 데이터를 읽어서, item-item graph를 만든다.
row_connect_user_to_item_dataframe = user_item.groupby('user_id', as_index=False)['item_id'].agg(lambda x: list(sorted(x)))
row_connect_user_to_item_dataframe['item_count'] = row_connect_user_to_item_dataframe['item_id'].apply(lambda x: len(x))

row_connect_item_to_user_dataframe = user_item.groupby('item_id', as_index=False)['user_id'].agg(lambda x: list(sorted(x)))
row_connect_item_to_user_dataframe['user_count'] = row_connect_item_to_user_dataframe['user_id'].apply(lambda x: len(x))

row_connect_user_to_item_dataframe.insert(1, 'u_id', row_connect_user_to_item_dataframe['user_id'])
row_connect_user_to_item_dataframe.set_index('u_id', inplace=True)

row_connect_item_to_user_dataframe.insert(1, 'i_id', row_connect_item_to_user_dataframe['item_id'])
row_connect_item_to_user_dataframe.set_index('i_id', inplace=True)

# train에서 1번도 출현하지 않은 itemset index에 대해 row 삽입
for i in list(set(range(total_number_user)) - set(row_connect_user_to_item_dataframe['user_id'])):
    row_connect_user_to_item_dataframe.loc[i] = [i, [], 0]
row_connect_user_to_item_dataframe = row_connect_user_to_item_dataframe.sort_index()

# train에서 1번도 출현하지 않은 itemset index에 대해 row 삽입
for i in list(set(range(total_number_item)) - set(row_connect_item_to_user_dataframe['item_id'])):
    row_connect_item_to_user_dataframe.loc[i] = [i, [], 0]
row_connect_item_to_user_dataframe = row_connect_item_to_user_dataframe.sort_index()

ui_user_sim_check = list(row_connect_user_to_item_dataframe.itertuples(index=False))
ui_item_sim_check = list(row_connect_item_to_user_dataframe.itertuples(index=False))

ui_user_sim_check.sort(key=lambda x: x[0])
ui_item_sim_check.sort(key=lambda x: x[0])
connection_dict_jaccard_sim = defaultdict(float)

item_value = defaultdict(float)
for item_data in tqdm(ui_user_sim_check):
    for i in range(len(item_data[1])):
        for j in range(i+1, len(item_data[1])):
            item_value[(ui_item_sim_check[item_data[1][i]][0], ui_item_sim_check[item_data[1][j]][0])] = jaccard_similarity(set(ui_item_sim_check[item_data[1][i]][1]), set(ui_item_sim_check[item_data[1][j]][1]))

with open(path + "/item_to_item_its_jac_with_user_item.csv", "w") as w:
    for (i, j), score in item_value.items():
        w.write("%d,%d,%f\n" %(i, j, score))