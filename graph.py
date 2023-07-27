import pandas as pd
import numpy as np
from scipy import stats
from functools import cmp_to_key
from collections import defaultdict

def graph_process(config_path):
    """
    Process the graph configuration file.

    Args:
        config_path (str): Path to the graph configuration file.

    Returns:
        tuple: A tuple containing start nodes, end nodes, partial order, and cluster unify.
    """
    cluster_unify = []
    partial_order = []
    start_candidates = []
    end_candidates = []
    
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                unify = list(map(int, line.split('=')))
                cluster_unify.append(unify)
            elif '<' in line:
                order = list(map(int, line.split('<')))
                partial_order.append(order)
                start_candidates.append(order[0])
                end_candidates.append(order[1])
    
    start = [element for element in start_candidates if element not in end_candidates]
    end = [element for element in end_candidates if element not in start_candidates]
    
    return list(set(start)), list(set(end)), partial_order, cluster_unify

class Graph: 
    """
    Class representing a graph.
    """
    def __init__(self, vertices): 
        self.V = vertices
        self.graph = defaultdict(list)  

    def addEdge(self, u, v): 
        """
        Add an edge to the graph.

        Args:
            u (int): Source vertex.
            v (int): Destination vertex.
        """
        self.graph[u].append(v) 

    def printPathsFunc(self, u, d, visited, path, current_path_list): 
        """
        Utility function for finding all paths between two vertices.

        Args:
            u (int): Current vertex.
            d (int): Destination vertex.
            visited (list): List of visited vertices.
            path (list): Current path being explored.
            current_path_list (list): List to store all paths found so far.

        Returns:
            list: List of all paths from u to d.
        """
        visited[u] = True
        path.append(u)

        if u == d: 
            path_copy = path[:]
            current_path_list.append(path_copy)
        else: 
            for i in self.graph[u]: 
                if not visited[i]: 
                    self.printPathsFunc(i, d, visited, path, current_path_list) 

        path.pop()
        visited[u] = False
        return current_path_list

    def printPaths(self, s, d): 
        """
        Find all paths between start nodes and end nodes.

        Args:
            s (list): List of start nodes.
            d (list): List of end nodes.

        Returns:
            list: List of all paths between start nodes and end nodes.
        """
        total_results = []
        for start in s:
            for dest in d:
                path = []
                visited = [False] * self.V
                current_path_list = []
                current_path_results = self.printPathsFunc(start, dest, visited, path, current_path_list) 
                if len(current_path_results) != 0:
                    total_results.extend(current_path_results)
        return total_results

def graph_inference_nightlight(grid_df, nightlight_df, cluster_num, file_path):
    """
    Perform graph inference based on nightlight data.

    Args:
        grid_df (DataFrame): DataFrame with grid data.
        nightlight_df (DataFrame): DataFrame with nightlight data.
        cluster_num (int): Number of clusters.
        file_path (str): Path to save the graph configuration file.

    Returns:
        list: List of ordered clusters based on nightlight data.
    """
    def numeric_compare(x, y):
        pop_list1 = df_merge_group.get_group(x)['nightlights'].tolist()
        pop_list2 = df_merge_group.get_group(y)['nightlights'].tolist()
        tTestResult = stats.ttest_ind(pop_list1, pop_list2)
        if (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) < np.mean(pop_list2)):
            return 1
        elif (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) >= np.mean(pop_list2)):
            return -1
        else:
            return 0
        
    df_merge = pd.merge(nightlight_df, grid_df, how='left', on='y_x')
    df_merge = df_merge.dropna()
    df_merge_group = df_merge.groupby('cluster_id')
    
    sorted_list = sorted(range(cluster_num - 1), key=cmp_to_key(numeric_compare))
    ordered_list = []
    ordered_list.append([sorted_list[0]])
    curr = 0
    for i in range(len(sorted_list) - 1):
        if numeric_compare(sorted_list[i], sorted_list[i+1]) == 0:
            ordered_list[curr].append(sorted_list[i+1])
        else:
            curr += 1
            ordered_list.append([sorted_list[i+1]])
            
    ordered_list.append([cluster_num - 1])        
    save_graph_config(ordered_list, file_path)
    return ordered_list

def save_graph_config(ordered_list, name):
    """
    Save the graph configuration to a file.

    Args:
        ordered_list (list): List of ordered clusters.
        name (str): Name of the configuration file.
    """
    with open(name, 'w') as f:
        for i in range(len(ordered_list) - 1):
            f.write('{}<{}\n'.format(ordered_list[i+1][0], ordered_list[i][0]))
        
        for orders in ordered_list:
            if len(orders) >= 2:
                f.write(str(orders[0]))
                for element in orders[1:]:
                    f.write('={}'.format(element))
                f.write('\n')