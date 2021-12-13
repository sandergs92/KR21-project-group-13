from typing import Union
from BayesNet import BayesNet
from collections import defaultdict
import networkx as nx


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

    # TODO: This is where your methods should go
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
        # Create interaction graph and set variable with nodes
        interaction_graph = self.bn.get_interaction_graph()
        x_vars = interaction_graph.nodes()
        # Create copy of interaction graph without links between nodes
        min_fill_graph = interaction_graph.__class__()
        min_fill_graph.add_nodes_from(interaction_graph)
        # dictionary to store results in
        min_fill_order = {}
        # Loop through each node and retrieve its neighbours
        for node in x_vars:
            neighbours = list(interaction_graph.neighbors(node))
            # Loop over neighbours and check if they are are connected
            for x in neighbours:
                for y in neighbours:
                    # print(x, y)
                    if x == y:
                        continue
                    # if neighbours are not connected, create new link between neighbours
                    elif x not in interaction_graph.neighbors(y):
                        min_fill_graph.add_edge(x, y)
        # loop over all nodes and count amount of new connections
        fill_vars = min_fill_graph.nodes()
        for node in fill_vars:
            var_edges = min_fill_graph.edges(node)
            min_fill_order[node] = len(var_edges)
        # remove node with least amount of filled links from interaction graph
        min_fill_node = min(min_fill_order, key=min_fill_order.get)
        interaction_graph.remove_node(min_fill_node)
        return sorted(min_fill_order.items(), key=lambda x: x[1], reverse=True)

    def prune_nodes(self, query, evidence):
        pass