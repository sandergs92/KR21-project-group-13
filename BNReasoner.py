from typing import Union
from BayesNet import BayesNet
from collections import defaultdict
from itertools import product, chain, combinations
import networkx as nx
import pandas as pd
import numpy as np
import math
import copy

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

    def d_seperation(self, x: list[str], y: list[str], givens: list[str]):
        # ANCESTRAL GRAPH
        # First create subgraph from given variables
        nodes = x + y + givens
        ancestral_graph = self.prune_network(nodes)
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
        nx.draw(undirected_ancestral_graph, with_labels = True)
        result_dsep = False
        for x_node, y_node in product(x, y):
            if nx.has_path(undirected_ancestral_graph, x_node, y_node):
                result_dsep = False
            else:
                result_dsep = True
        return result_dsep

    def min_degree_order(self, elim_vars: list[str]):
        # Create interaction graph and set variable with nodes
        interaction_graph = copy.deepcopy(self.bn.get_interaction_graph())
        min_degree_order = []
        # Loop through amount of variables/nodes
        for _ in range(0, len(elim_vars)):
            # Loop through each node, make dict of amount neighbours
            neigbour_dict = defaultdict(int)
            for node in elim_vars:
                if node not in min_degree_order:
                    neigbour_dict[node] = len(list(interaction_graph.neighbors(node)))
            # Select node that minimizes |ne(X)| and add it to the order
            min_neigbours_node = min(neigbour_dict, key=neigbour_dict.get)
            min_degree_order.append(min_neigbours_node)
            # Add an edge between each non--adjacent neighbours
            for node in interaction_graph.nodes():
                neighbors = list(interaction_graph.neighbors(node))
                for previous, current in zip(neighbors, neighbors[1:]):
                    interaction_graph.add_edge(previous, current)
            # remove node from interaction graph
            interaction_graph.remove_node(min_neigbours_node)
        return min_degree_order

    def min_fill_order(self, elim_vars: list[str]):
        # Create interaction graph and set variable with nodes
        interaction_graph = copy.deepcopy(self.bn.get_interaction_graph())
        min_fill_order = []
        # Loop through each node and retrieve its neighbours
        for _ in range(0, len(elim_vars)):
            # Loop through each node, make dict of amount neighbours
            fill_in_dict = {}
            for node in elim_vars:
                if node not in min_fill_order:
                    # Get neighbors of node
                    neighbors = list(interaction_graph.neighbors(node))
                    fill_in_dict[node] = {'n_edges': 0, 'neighbors': []}
                    if len(neighbors) > 1:
                        comb_neighbors = list(combinations(neighbors, r=2)) 
                        fill_in_dict[node]['neighbors'] += [edge for edge in comb_neighbors if edge not in interaction_graph.edges]
                        fill_in_dict[node]['n_edges'] = len(fill_in_dict[node]['neighbors'])
            min_fill_in_node = min(fill_in_dict, key=lambda x: fill_in_dict[x]['n_edges'])
            min_fill_order.append(min_fill_in_node)
            # Add an edge between each non--adjacent neighbours
            interaction_graph.add_edges_from(fill_in_dict[min_fill_in_node]['neighbors'])
            # remove node from interaction graph
            interaction_graph.remove_node(min_fill_in_node)
        return min_fill_order

    def get_joint_probability_distribution(self):
        cpt_list = [self.bn.get_cpt(x) for x in self.bn.get_all_variables()] 
        joint_probability_distribution = self.multiply_factors(cpt_list)
        return joint_probability_distribution

    def create_empty_truth_table(self, cpt_vars: list[str]):
        n_vars = len(cpt_vars)
        cpt_vars.append('p')
        empty_cpt = pd.DataFrame(columns=cpt_vars, index=range(2**(n_vars)))
        # Fill table with truth values
        truth_values = [list(i) for i in product([False, True], repeat=n_vars)]
        for i in range(2**(n_vars)):
            empty_cpt.loc[i] = truth_values[i] + [np.nan]
        return empty_cpt

    def max_out_vars(self, cpt: pd.DataFrame, subset_vars: list[str]):
        maxxed_out_cpt = cpt.drop(subset_vars, axis=1)
        new_vars = [item for item in cpt.columns.tolist()[:-1] if item not in subset_vars]
        maxxed_out_cpt = maxxed_out_cpt.groupby(new_vars).max().reset_index()
        return maxxed_out_cpt

    def sum_out_vars(self, cpt: pd.DataFrame, subset_vars: list[str]):
        summed_out_cpt = cpt.drop(subset_vars, axis=1)
        new_vars = [item for item in cpt.columns.tolist()[:-1] if item not in subset_vars]
        summed_out_cpt = summed_out_cpt.groupby(new_vars).sum().reset_index()
        return summed_out_cpt

    def multiply_factors(self, cpts: list[pd.DataFrame]):
        if len(cpts) == 1:
            return cpts[0]
        final_cpt = 0
        for previous, current in zip(cpts, cpts[1:]):
            if not isinstance(final_cpt, pd.DataFrame):
                final_cpt = previous
            final_cpt = self.cpt_product(current, final_cpt)
        return final_cpt

    def cpt_product(self, cpt_1: pd.DataFrame, cpt_2: pd.DataFrame):
        # Create new table
        column_cpt1 = cpt_1.columns.tolist()[:-1]
        column_cpt2 = cpt_2.columns.tolist()[:-1]
        new_column = column_cpt2 + list(set(column_cpt1) - set(column_cpt2))
        cpt_product = self.create_empty_truth_table(new_column)
        # Iterate through each row of new CPT
        iters = [cpt_1.iterrows(), cpt_2.iterrows(), cpt_product.iterrows()]
        for row_cpt1, row_cpt2, row_new_cpt in product(*iters):
            if row_cpt1[1][:-1].to_dict().items() <= row_new_cpt[1][:-1].to_dict().items() and row_cpt2[1][:-1].to_dict().items() <= row_new_cpt[1][:-1].to_dict().items():
                if math.isnan(cpt_product.iloc[[row_new_cpt[0]]]['p']):
                    result = row_cpt1[1]['p'] * row_cpt2[1]['p']
                    cpt_product.at[row_new_cpt[0], 'p'] = result
        return cpt_product

    def prior_marginal(self, query_vars: list[str]):
        ## first multiply every variable from query
        start_cpt = self.multiply_factors([self.bn.get_cpt(node) for node in query_vars])
        # BRAIN FART
        # maak subgraph van ancestors
        # bepaal het pad has path
        ancestors = [list(nx.algorithms.dag.ancestors(self.bn.structure, node)) for node in query_vars]
        # order heuristic needs to be dynamic
        pi = self.min_degree_order(list(chain(*ancestors)))
        pi.reverse()

        for node in pi: 
            # Multiply
            start_cpt = self.cpt_product(start_cpt, self.bn.get_cpt(node))
            # Sum-out
            start_cpt = self.sum_out_vars(start_cpt, [node])
        return start_cpt

    def node_pruning(self, rest_nodes: list[str]):
        subgraph = self.bn.structure.subgraph(rest_nodes).copy()
        # Then add all ancestors of given variables
        for node in rest_nodes:
            # Find out ancestors
            ancestors = list(nx.algorithms.dag.ancestors(self.bn.structure, node))
            ancestors.append(node)
            ancestors_subgraph = self.bn.structure.subgraph(ancestors).copy()
            subgraph = nx.algorithms.operators.binary.compose(subgraph, ancestors_subgraph)
        return subgraph

    def edge_pruning(self, node_pruned_network: nx.Graph, evidence: list[tuple]):
        cpts = self.bn.get_all_cpts()
        copy_pruned_network = node_pruned_network.copy()
        # Drop all cpts that aren't relevant
        for k in self.bn.get_all_variables():
            if k not in list(node_pruned_network.nodes()):
                cpts.pop(k, None)
        for piece in evidence:
            for edge in node_pruned_network.edges(piece):
                cpt = cpts[edge[1]]
                # Drop rows and columns according to evidence
                indexNames = cpt[cpt[piece[0]] == (not piece[1])].index
                cpt = cpt.drop(indexNames, inplace=False).reset_index(drop=True)
                cpt = cpt.drop([piece[0]], axis=1, inplace=False)
                cpts[edge[1]] = cpt
                # Remove edge
                copy_pruned_network.remove_edge(edge[0], edge[1])
        return copy_pruned_network, cpts

    def prune_network(self, query: list[str], evidence: list[tuple] = None):
        node_pruned_network = self.node_pruning(query)
        if evidence:
            pruned_network, cpts = self.edge_pruning(node_pruned_network, evidence)
            return pruned_network, cpts
        return node_pruned_network