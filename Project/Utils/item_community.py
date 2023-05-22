from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import networkx as nx
import itertools

class community:
    def __init__(self,
        file_path='./dataset/ii_item_pair_weight.csv',
        weight=False,
        preprocessed=True):
        """
        Make the class 'community' for the hierarchical community detection.
        
        Parameters
        ----------
        file_path : str, default = './dataset/ii_item_pair_weight.csv'
            Enter the path where the file that contains all pairs among all items is located.

        weight : bool, default = False
            If you want to construct weighted graph, set this parameter to True
            
        preprocessed : bool, default = True
            If already preprocessed, skip _make_dataframe
        
        """
        self.file_path = file_path
        self.total_number_itemset = 27694
        self.weight = weight
        self.df = self._make_dataframe(self.file_path) if not preprocessed else self._preprocessed_dataframe(self.file_path)
        if not self.weight:
            print('Class Community : Convert weighted to unweighted')
            self.df['weight'] = 1
        self.G = self._make_graph()

    def make_hierarchical_community(self, step=10, threshold=1):
        """
        Make the hierarchical community detection with graph, using louvain.
        
        Parameters
        ----------
        step : int, default = 10
            Determine how far the depth will go. Iteration may be terminated automatically before that.
        threshold : int, default = 10
            Determine if the communities exist equal or less than threshold, Terminated.
        
        Returns
        -------
        louvain_community_list : list[list[int]]
            Louvain community information at each step
            0th index n: n-th step of louvain (n starts with 0)
            1st index m: m-th community infomation in n-th step of louvain (m starts with 0)

        hierarchical_community_data : defaultdict(list[int])
            Community number to which a specific node in each step belongs
            0th index n: the node number (n starts with 0)
            1st index m: the community number of the n-th node in m-th step of louvain (m starts with 0) 
        """
        if self.weight:
            G_louvain = nx.community.louvain_partitions(self.G, weight='weight', resolution=0.33, threshold=0)
        else:
            G_louvain = nx.community.louvain_partitions(self.G, resolution=0.33, threshold=0)

        # save data each iteration of louvain_partitions
        print('Class Community : Make the louvain community data in each step')
        louvain_community_list = []
        for communities in itertools.islice(G_louvain, step):
            louvain_community_list.append(list(sorted(c) for c in communities))
        
        for index in range(len(louvain_community_list)):
            louvain_community_list[index].sort(key=lambda x: len(x), reverse=True)

        # make hierarchical community dictionary
        hierarchical_community_data = defaultdict(list)
        
        # check recently community of each item -> matching
        for i in range(len(louvain_community_list)):
            total_live_community = set()
            print(f'Class Community : Make the community step {i}')
            print(f'Class Community : The size of the 5 largest community in step {i} : [{len(louvain_community_list[i][0])}, {len(louvain_community_list[i][1])}, {len(louvain_community_list[i][2])}, {len(louvain_community_list[i][3])}, {len(louvain_community_list[i][4])}]')
            for index, l in tqdm(enumerate(louvain_community_list[i])):
                for item in l:
                    # if i != 0:
                    #     print(len(louvain_community_list[i-1]), index, item, hierarchical_community_data[item][i-1])
                    if i != 0 and (hierarchical_community_data[item][i-1] is None or 
                                   len(louvain_community_list[i][index]) == len(louvain_community_list[i-1][hierarchical_community_data[item][i-1]])):
                        hierarchical_community_data[item].append(None)
                    else:
                        hierarchical_community_data[item].append(index)
            for j in range(len(hierarchical_community_data)):
                if hierarchical_community_data[j][-1] is None:
                    continue
                total_live_community.add(hierarchical_community_data[j][-1])
            print(f'Class Community : Total community in {i} iteration -> {len(total_live_community)}')
            if len(total_live_community) <= threshold:
                break

        return louvain_community_list, hierarchical_community_data

    def _make_dataframe(self, file_path):
        """
        Make the pd.DataFrame from the file.
        
        Parameters
        ----------
        file_path : str, default = './dataset/itemset_item_training.csv'
            Enter the path where the file 'itemset_item_training.csv' is located.
        
        """
        print('Class Community : Load the itemset_item_training.csv')
        itemset_item_training = pd.read_csv(file_path, delimiter=',', names=['itemset_id', 'item_id'])
        row_connect_itemset_to_item_dataframe = itemset_item_training.groupby('itemset_id', as_index=False)['item_id'].agg(lambda x: list(sorted(x)))
        row_connect_itemset_to_item_dataframe['item_count'] = row_connect_itemset_to_item_dataframe['item_id'].apply(lambda x: len(x))

        row_connect_item_to_itemset_dataframe = itemset_item_training.groupby('item_id', as_index=False)['itemset_id'].agg(lambda x: list(sorted(x)))
        row_connect_item_to_itemset_dataframe['itemset_count'] = row_connect_item_to_itemset_dataframe['itemset_id'].apply(lambda x: len(x))

        row_connect_itemset_to_item_dataframe.insert(1, 'iset_id', row_connect_itemset_to_item_dataframe['itemset_id'])
        row_connect_itemset_to_item_dataframe.set_index('iset_id', inplace=True)

        for i in list(set(range(self.total_number_itemset)) - set(row_connect_itemset_to_item_dataframe['itemset_id'])):
            row_connect_itemset_to_item_dataframe.loc[i] = [i, [], 0]
        row_connect_itemset_to_item_dataframe = row_connect_itemset_to_item_dataframe.sort_index()

        ii_itemset_sim_check = list(row_connect_itemset_to_item_dataframe.itertuples(index=False))
        ii_item_sim_check = list(row_connect_item_to_itemset_dataframe.itertuples(index=False))

        ii_itemset_sim_check.sort(key=lambda x: x[0])
        ii_item_sim_check.sort(key=lambda x: x[0])

        df = pd.DataFrame(columns=['item_id1', 'item_id2', 'weight'])
        print('Class Community : Make the similarity map')
        for itemset_data in tqdm(ii_itemset_sim_check):
            for i in range(len(itemset_data[1])):
                for j in range(i+1, len(itemset_data[1])):
                    df.loc[len(df)] = [ii_item_sim_check[itemset_data[1][i]][0], 
                                       ii_item_sim_check[itemset_data[1][j]][0],
                                       self._similarity(set(ii_item_sim_check[itemset_data[1][i]][1]), set(ii_item_sim_check[itemset_data[1][j]][1]))]
        return df
    
    def _preprocessed_dataframe(self, file_path):
        """
        Make the pd.DataFrame from the preprocessed file.
        
        Parameters
        ----------
        file_path : str
            Enter the path where the preprocessed file is located.
        
        """    
        print('Class Community : Load the preprocessed csv file')
        df = pd.read_csv(file_path, delimiter=',', names=['item_id1', 'item_id2', 'weight'])
        return df

    def _make_graph(self):
        """
        Make the networkx graph from pd.DataFrame.
        
        Parameters
        ----------
        None

        """
        print('Class Community : Convert DataFrame to networkx graph')
        G = nx.from_pandas_edgelist(self.df, 'item_id1', 'item_id2', ['weight'])
        G = G.to_undirected()
        return G
    
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
        return similarity