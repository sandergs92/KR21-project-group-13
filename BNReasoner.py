from typing import Union
from BayesNet import BayesNet
from collections import defaultdict
from itertools import product, combinations, groupby
import networkx as nx
import pandas as pd
import numpy as np
import math
import copy
import random

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
        ancestral_graph, cpts = self.prune_network(nodes)
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
        undirected_ancestral_graph.remove_nodes_from(givens)
        nx.draw(undirected_ancestral_graph, with_labels = True)
        result_dsep = False
        for x_node, y_node in product(x, y):
            if nx.has_path(undirected_ancestral_graph, x_node, y_node):
                result_dsep = False
            else:
                result_dsep = True
        return result_dsep

    def random_order(self, elim_vars: list[str]):
        random_order = elim_vars
        random.shuffle(random_order)
        return random_order

    def min_degree_order(self, elim_vars: list[str]):
        # Create interaction graph and set variable with nodes
        interaction_graph = copy.deepcopy(self.bn.get_interaction_graph())
        min_degree_order = []
        # Loop through amount of variables/nodes
        for _ in range(len(elim_vars)):
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
        for _ in range(len(elim_vars)):
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

    def posterior_marginal(self, query_vars: list[str], evidence: list[tuple] = None, elimination_heuristic: int = 1):
        pruned_network, pruned_cpts = self.prune_network(query_vars, evidence)
        joint_marginal = self.joint_marginal(query_vars, evidence, elimination_heuristic=elimination_heuristic)
        evidence_factor = 1.0
        for piece in evidence:
            evidence_factor *= float(pruned_cpts[piece[0]].loc[pruned_cpts[piece[0]][piece[0]] == piece[1], 'p'])
        joint_marginal['p'] = joint_marginal['p'].div(evidence_factor)
        return joint_marginal

    def joint_marginal(self, query_vars: list[str], evidence: list[tuple], elimination_heuristic: int = 1):
        evidence_vars = [x[0] for x in evidence]
        prior_marginal_cpt = self.prior_marginal(query_vars + evidence_vars, elimination_heuristic=elimination_heuristic)
        for piece in evidence:
            prior_marginal_cpt = prior_marginal_cpt.loc[prior_marginal_cpt[piece[0]] == piece[1]]
        prior_marginal_cpt = prior_marginal_cpt.drop(evidence_vars, axis=1, inplace=False)
        prior_marginal_cpt = prior_marginal_cpt.reset_index(drop=True)
        return prior_marginal_cpt

    def prior_marginal(self, query_vars: list[str], elimination_heuristic: int = 1):
            # First prune network
            pruned_network, pruned_cpts = self.prune_network(query_vars)
            # Determine elimination order
            if elimination_heuristic == 1:
                var_elim_order = self.min_degree_order(pruned_network.nodes() - query_vars)
            elif elimination_heuristic == 2:
                var_elim_order = self.min_fill_order(pruned_network.nodes() - query_vars)
            # Loop through each eliminated node, remove them 
            for elim_node in var_elim_order:
                for node in pruned_network.nodes():
                    if elim_node == node or node not in pruned_cpts.keys():
                        continue
                    elim_node_cpt = pruned_cpts[elim_node]
                    network_node_cpt = pruned_cpts[node]
                    # Check for each CPT, does it contain the variable needs to be eliminated
                    if elim_node in network_node_cpt.columns.tolist()[:-1]:
                        # if found variable elimination
                        result_cpt = self.cpt_product(elim_node_cpt, network_node_cpt)
                        result_cpt = self.sum_out_vars(result_cpt, [elim_node])
                        pruned_cpts[node] = result_cpt
                # If we finished for loop, pop elimination cpt from cpts
                pruned_cpts.pop(elim_node, None)
            # Lastly multiply each new factor to get the prior marginal 
            return self.multiply_factors(list(pruned_cpts.values()))

    def mpe(self, evidence: list[tuple], elimination_heuristic: int = 1):
        # Prune network given the evidence
        pruned_network, pruned_cpts = self.edge_pruning(self.bn.structure, evidence, self.bn.get_all_cpts())
        q_nodes = list(pruned_network)
        if elimination_heuristic == 1:
                var_elim_order = self.min_degree_order(q_nodes)
        elif elimination_heuristic == 2:
            var_elim_order = self.min_fill_order(q_nodes)
        elif elimination_heuristic == 3:
            var_elim_order = self.random_order(q_nodes)
        for elim_node, node in product(var_elim_order, pruned_network.nodes()):
            if elim_node == node or node not in pruned_cpts.keys() or elim_node not in pruned_cpts.keys():
                continue
            elim_node_cpt = pruned_cpts[elim_node]
            network_node_cpt = pruned_cpts[node]
            # Check for each CPT, does it contain the variable needs to be eliminated
            if elim_node in network_node_cpt.columns.tolist()[:-1]:
                result_cpt = self.cpt_product(elim_node_cpt, network_node_cpt)
                result_cpt = self.max_out_vars(result_cpt, [elim_node])
                pruned_cpts[node] = result_cpt
        mpe_cpt = self.multiply_factors(list(pruned_cpts.values()))
        if evidence: 
            query = ''
            for idx, piece in enumerate(evidence):
                query += '`' + piece[0] + '` == ' + str(piece[1])
                if idx != (len(evidence) - 1):
                    query += ' & '
            mpe_cpt = mpe_cpt.query(query)
        return mpe_cpt[mpe_cpt['p']==mpe_cpt['p'].max()]

    def map_instance(self, query_vars: list[str], evidence: list[tuple], elimination_heuristic: int = 1):
        # prune network 
        pruned_network, pruned_cpts = self.prune_network(query_vars, evidence)
        # var elim order 
        evidence_vars = [x[0] for x in evidence]
        if elimination_heuristic == 1:
            var_elim_order = self.min_degree_order(evidence_vars)
            var_elim_order += self.min_degree_order(query_vars)
        elif elimination_heuristic == 2:
            var_elim_order = self.min_fill_order(evidence_vars)
            var_elim_order += self.min_fill_order(query_vars)
        elif elimination_heuristic == 3:
            var_elim_order = self.random_order(evidence_vars)
            var_elim_order += self.random_order(query_vars)
        for elim_node, node in product(var_elim_order, pruned_network.nodes()):
            if elim_node == node or node not in pruned_cpts.keys() or elim_node not in pruned_cpts.keys():
                continue
            elim_node_cpt = pruned_cpts[elim_node]
            network_node_cpt = pruned_cpts[node]
            if elim_node in network_node_cpt.columns.tolist()[:-1]:
                result_cpt = self.cpt_product(elim_node_cpt, network_node_cpt)
                if elim_node in query_vars:
                    result_cpt = self.max_out_vars(result_cpt, [elim_node])
                else:
                    result_cpt = self.sum_out_vars(result_cpt, [elim_node])
                pruned_cpts[node] = result_cpt
        map_cpt = self.multiply_factors(list(pruned_cpts.values()))
        try:
            if evidence:
                query = ''
                for idx, piece in enumerate(evidence):
                    query += '`' + piece[0] + '` == ' + str(piece[1])
                    if idx != (len(evidence) - 1):
                        query += ' & '
                map_cpt = map_cpt.query(query)
        except:
            print(".")
        return map_cpt[map_cpt['p']==map_cpt['p'].max()]

    def node_pruning(self, rest_nodes: list[str]):
        cpts = self.bn.get_all_cpts()
        subgraph = self.bn.structure.subgraph(rest_nodes).copy()
        # Then add all ancestors of given variables
        for node in rest_nodes:
            # Find out ancestors
            ancestors = list(nx.algorithms.dag.ancestors(self.bn.structure, node))
            ancestors.append(node)
            ancestors_subgraph = self.bn.structure.subgraph(ancestors).copy()
            subgraph = nx.algorithms.operators.binary.compose(subgraph, ancestors_subgraph)
        # Drop all cpts that aren't relevant
        for node in self.bn.get_all_variables():
            if node not in list(subgraph.nodes()):
                cpts.pop(node, None)
        return subgraph, cpts

    def edge_pruning(self, node_pruned_network: nx.Graph, evidence: list[tuple], pruned_cpts: dict):
        copy_cpts = pruned_cpts
        copy_pruned_network = node_pruned_network.copy()
        for piece in evidence:
            for edge in node_pruned_network.edges(piece):
                cpt = copy_cpts[edge[1]]
                # Drop rows and columns according to evidence
                indexNames = cpt[cpt[piece[0]] == (not piece[1])].index
                cpt = cpt.drop(indexNames, inplace=False).reset_index(drop=True)
                cpt = cpt.drop([piece[0]], axis=1, inplace=False)
                copy_cpts[edge[1]] = cpt
                # Remove edge
                copy_pruned_network.remove_edge(edge[0], edge[1])
        return copy_pruned_network, copy_cpts

    def prune_network(self, query: list[str], evidence: list[tuple] = None):
        node_pruned_network, pruned_cpts = self.node_pruning(query)
        if evidence:
            pruned_network, pruned_cpts = self.edge_pruning(node_pruned_network, evidence, pruned_cpts)
            return pruned_network, pruned_cpts
        return node_pruned_network, pruned_cpts

    # [code adapted from: https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx]
    # [author: yatu]
    # [taken on: 16/12/2021]
    ### Make sure to store random graphs and cpts that we use for our experiment!
    def gnp_random_connected_graph(self, size=10, probability=0.1):
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi 
        graph, but enforcing that the resulting graph is connected
        """
        edges = combinations(range(size), 2)
        G = nx.DiGraph()
        node_names = ['node_' + str(x) for x in range(size)]
        G.add_nodes_from(node_names)
        if probability <= 0:
            return G
        if probability >= 1:
            return nx.complete_graph(size, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            node_edges = [('node_' + str(edge[0]), 'node_' + str(edge[1])) for edge in node_edges]
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
            for e in node_edges:
                if random.random() < probability:
                    G.add_edge(*e)
        return G

    def set_cpts(self, network: nx.DiGraph):
        # retrieve nodes in the system
        nodes = list(network.nodes())
        cpts = {}
        for node in nodes:
            # retrieve parents of each node
            predecessors = list(network.predecessors(node))
            # create empty truth tables for each node
            empty_truth_table = self.create_empty_truth_table(predecessors + [node])
            # Assign to dictionary, using node as key and empty truth table as value
            cpts[node] = empty_truth_table
        # iterate over dictionary and assign probabilities
        for _, t_table in cpts.items():
            table_size = len(t_table)
            for i in range(0, table_size, 2):
                prob = round(random.uniform(0, 1), 2)
                true_value = 1 - prob
                t_table.at[t_table.index[i], 'p'] = prob
                t_table.at[t_table.index[i+1], 'p'] = true_value
        return cpts

    def performance_evaluation(self, n_networks, factor=10, size=10, probability=0.1):
        network_dict = {}
        # create n_networks number of randomized Bayesian networks
        count = 1
        for _ in range(n_networks):
            name = "network" + str(count)
            network = self.gnp_random_connected_graph(size, probability)
            cpts = self.set_cpts()
            network_dict[name] = {'network': network, 'cpts': cpts}
            count += 1
            size += factor
        return network_dict