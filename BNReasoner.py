from typing import Union
from BayesNet import BayesNet
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