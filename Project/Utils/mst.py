from collections import defaultdict
from tqdm import tqdm

import pandas as pd

class MST:
    def __init__(self, data='./Dataset/item_to_item_its_jac_with_user_item.csv'):
        """
        Make the class 'MST' for the extended style graph.
        
        Parameters
        ----------
        data : str, default = './Dataset/item_to_item_its_jac_with_user_item.csv'
            Enter the path where the file that contains all pairs among all items is located.
        
        """
        self.parent = defaultdict(int)
        self.rank = defaultdict(int)
        self.graph = defaultdict(list)
        self.graph['vertices'] = [i for i in range(42563)]
        
        print('Class MST: Load the item_to_item_its_jac_with_user_item.csv')
        with open(data, 'r') as f:
            for row in tqdm(f.readlines()):
                i1, i2, w = map(float, row.strip().split(','))
                i1, i2 = int(i1), int(i2)
                self.graph['edges'].append((i1, i2, w))

    def kruskal(self) -> list: 
        """
        Make the pd.DataFrame from the file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        edge_data : defaultdict(float)
            Similarities between two nodes expressed as real numbers
            Each key pair is the item pair (item1, item2)

        """
        def make_set(vertice: int) -> None:
            self.parent[vertice] = vertice
            self.rank[vertice] = 0

        def find(vertice: int) -> int:
            if self.parent[vertice] != vertice:
                self.parent[vertice] = find(self.parent[vertice])
            return self.parent[vertice]

        def union(vertice1: int, vertice2: int) -> None:
            root1: int = find(vertice1)
            root2: int = find(vertice2)
            if root1 != root2:
                if self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                else:
                    self.parent[root1] = root2
                    if self.rank[root1] == self.rank[root2]: 
                        self.rank[root2] += 1

        maximum_spanning_tree: list = []

        for vertice in self.graph['vertices']:
            make_set(vertice)
            
        edges: list = self.graph['edges']
        edges.sort(key = lambda x: x[2], reverse=True)
        
        for edge in tqdm(edges):
            vertice1, vertice2, _ = edge
            if find(vertice1) != find(vertice2):
                union(vertice1, vertice2)
                maximum_spanning_tree.append(edge)
            
        return maximum_spanning_tree
    
    def extended_style_graph(self, data, file_path='./Dataset/itemset_item_training.csv'):
        """
        Make the extend style graph from the MST and itemset_item_training.csv.
        Before calling that, 'item_to_item_its_jac.csv' must exist in the 'Dataset folder'.

        Parameters
        ----------
        data: list
            Put the data of Maximum spanning Tree (Result of the MST.kruskal)

        file_path : str, default = './Dataset/itemset_item_training.csv'
            Enter the path where the file 'itemset_item_training.csv' is located.
        
        Returns
        -------
        edge_data : defaultdict(float)
            Similarities between two nodes expressed as real numbers
            Each key pair is the item pair (item1, item2)
            key : (item1, item2)
            value : score (item1 & item2 + 1) / (item1 | item2 + 1)

        """
        print('Class MST: Load the itemset_item_training.csv')
        itemset_item_training = pd.read_csv(file_path, delimiter=',', names=['itemset_id', 'item_id'])
        row_connect_itemset_to_item_dataframe = itemset_item_training.groupby('itemset_id', as_index=False)['item_id'].agg(lambda x: list(sorted(x)))
        row_connect_itemset_to_item_dataframe['item_count'] = row_connect_itemset_to_item_dataframe['item_id'].apply(lambda x: len(x))

        row_connect_item_to_itemset_dataframe = itemset_item_training.groupby('item_id', as_index=False)['itemset_id'].agg(lambda x: list(sorted(x)))
        row_connect_item_to_itemset_dataframe['itemset_count'] = row_connect_item_to_itemset_dataframe['itemset_id'].apply(lambda x: len(x))

        row_connect_itemset_to_item_dataframe.insert(1, 'iset_id', row_connect_itemset_to_item_dataframe['itemset_id'])
        row_connect_itemset_to_item_dataframe.set_index('iset_id', inplace=True)

        for i in list(set(range(27694)) - set(row_connect_itemset_to_item_dataframe['itemset_id'])):
            row_connect_itemset_to_item_dataframe.loc[i] = [i, [], 0]
        row_connect_itemset_to_item_dataframe = row_connect_itemset_to_item_dataframe.sort_index()

        ii_itemset_sim_check = list(row_connect_itemset_to_item_dataframe.itertuples(index=False))
        ii_item_sim_check = list(row_connect_item_to_itemset_dataframe.itertuples(index=False))

        ii_itemset_sim_check.sort(key=lambda x: x[0])
        ii_item_sim_check.sort(key=lambda x: x[0])

        print('Class MST: Make the extend style graph data')
        edge_data = defaultdict(float)
        with open('./Dataset/item_to_item_its_jac.csv') as f:
            for row in f.readlines():
                i, j, w = map(float, row.strip().split(','))
                i, j = int(i), int(j)
                edge_data[(i, j)] = w 
        for i, j, _ in data:
            edge_data[(i, j)] = self._similarity(set(ii_item_sim_check[i][1]), set(ii_item_sim_check[j][1]), metric='custom_jaccard')

        return edge_data
        
    def _similarity(self, s1, s2, metric='jaccard'):
        """
        Calculate Jaccard Similarity between two sets.

        Parameters
        ----------
        s1, s2 : set

        metric : Optional, str, default = 'jaccard'
            A measure of the similarity between two sets.

        Returns
        -------
        similarity : float
            Similarity between two sets expressed as real numbers

        """
        similarity = 0.0
        if metric == 'jaccard':
            similarity = len(s1 & s2) / len(s1 | s2)
        elif metric == 'overlap':
            similarity = len(s1 & s2) / min(len(s1), len(s2))
        elif metric == 'custom_jaccard':
            similarity = (len(s1 & s2) + 1) / (len(s1 | s2) + 1)
        return similarity