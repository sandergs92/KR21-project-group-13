from typing import Union
from BayesNet import BayesNet
from collections import defaultdict
from itertools import product
import networkx as nx
import pandas as pd
import numpy as np
import math

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def d_seperation(self, x, y, givens):
        # ANCESTRAL GRAPH
        # First create subgraph from given variables
        nodes = [x] + [y] + givens
        ancestral_graph = self.bn.structure.subgraph(nodes).copy()
        # Then add all ancestors of given variables
        for node in nodes:
            # Find out ancestors
            ancestors = list(nx.algorithms.dag.ancestors(self.bn.structure, node))
            ancestors_subgraph = self.bn.structure.subgraph(ancestors).copy()
            ancestral_graph = nx.algorithms.operators.binary.compose(ancestral_graph, ancestors_subgraph)
        # MORALIZE AND DISORIENT
        # First create undirected graph from ancestral graph
        undirected_ancestral_graph = ancestral_graph.to_undirected()
        # For each pair of variables with a common child, draw an undirected edge between them
        for node in ancestral_graph.nodes():
            predecessors = list(ancestral_graph.predecessors(node))
            if len(predecessors) > 1:
                for previous, current in zip(predecessors, predecessors[1:]):
                    undirected_ancestral_graph.add_edge(previous, current)
        # DELETE GIVENS
        for given in givens:
            undirected_ancestral_graph.remove_node(given)

        if nx.has_path(undirected_ancestral_graph, x, y):
            print("There is a path. Therefore, it is not guaranteed that they are independent.")
        else:
            print("There is no path between " + x + " and " + y + " given " + str(givens) + " , therefore they are independent.")
        nx.draw(undirected_ancestral_graph, with_labels = True)

    def min_degree_order(self):
        # Create interaction graph and set variable with nodes
        interaction_graph = self.bn.get_interaction_graph()
        x_vars = interaction_graph.nodes()
        min_degree_order = []
        # Loop through amount of variables/nodes
        for _ in range(0, len(x_vars)):
            # Loop through each node, make dict of amount neighbours
            neigbour_dict = defaultdict(int)
            for node in x_vars:
                neigbour_dict[node] = len(list(interaction_graph.neighbors(node)))
            # Select node that minimizes |ne(X)|
            min_neigbours_node = min(neigbour_dict, key=neigbour_dict.get)
            # add to order
            min_degree_order.append(min_neigbours_node)
            # remove node from interaction graph
            interaction_graph.remove_node(min_neigbours_node)
        return min_degree_order

    def min_fill_order(self):
        return 0

    def get_joint_probability_distribution(self):
        cpt_list = [self.bn.get_cpt(x) for x in self.bn.get_all_variables()] 
        joint_probability_distribution = self.multiply_factors(cpt_list)
        return joint_probability_distribution

    def multiply_factors(self, cpts: list):
        final_cpt = 0
        for previous, current in zip(cpts, cpts[1:]):
            if not isinstance(final_cpt, pd.DataFrame):
                final_cpt = previous
            final_cpt = self.cpt_product(current, final_cpt)
        return final_cpt

    def cpt_product(self, cpt_1, cpt_2):
        # Create new table
        column_cpt1 = cpt_1.columns.tolist()[:-1]
        column_cpt2 = cpt_2.columns.tolist()[:-1]
        new_column = column_cpt2 + list(set(column_cpt1) - set(column_cpt2))
        n_vars = len(new_column)
        new_column.append('p')
        cpt_product = pd.DataFrame(columns=new_column, index=range(2**(n_vars)))
        # Fill table with truth values
        truth_values = [list(i) for i in product([False, True], repeat=n_vars)]
        for i in range(2**(n_vars)):
            cpt_product.loc[i] = truth_values[i] + [np.nan]
        # Iterate through each row of new CPT
        iters = [cpt_1.iterrows(), cpt_2.iterrows(), cpt_product.iterrows()]
        for row_cpt1, row_cpt2, row_new_cpt in product(*iters):
            if row_cpt1[1][:-1].to_dict().items() <= row_new_cpt[1][:-1].to_dict().items() and row_cpt2[1][:-1].to_dict().items() <= row_new_cpt[1][:-1].to_dict().items():
                if math.isnan(cpt_product.iloc[[row_new_cpt[0]]]['p']):
                    result = row_cpt1[1]['p'] * row_cpt2[1]['p']
                    cpt_product.at[row_new_cpt[0], 'p'] = result
        return cpt_product