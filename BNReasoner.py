from typing import Union
from BayesNet import BayesNet
import networkx as nx
import pandas as pd

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


    def node_pruning(Query, evidence):
        """Prune the leaf nodes that are not within the union of Q and e. (not-recursive)"""
        dog_network_test = BayesNet()
        dog_network_test.load_from_bifxml('testing/dog_problem.BIFXML')
        leaf_nodes = []
        for_deleting = []
        
        # Gather all the nodes 
        all_nodes = dog_network_test.get_all_variables()
        print(f"ALL NODES: {all_nodes}")
        
        # Isolate the leaf nodes: nodes with no bebehs. Store them in leaf_nodes list.
        for node in all_nodes:
            if len(dog_network_test.get_children(node)) == 0:
                leaf_nodes.append(node)
        print(f"HERE ARE THE LEAF NODES: {leaf_nodes}")
        # Check if the isolated leaf nodes are part of Q and e combined. If not in Q and e: add them to for_deleting list
        for node in leaf_nodes:
            if node not in Query and evidence:
                for_deleting.append(node)
        print(f"NODES FOR DELETING: {for_deleting}")
    
        # return/draw the remaining nodes and edges.
        for n in for_deleting:
            dog_network_test.del_var(n)
        return dog_network_test.draw_structure()

        # Change the CPT Tables:
        CPT = dog_network_test.get_all_cpts()
        
# Q = ["family-out"]
# e = ["light-on", "dog-out"]
# test = BNReasoner
# test.node_pruning(Q, e)

    def edge_pruning(Query, evidence):
        """Get rid of the outgoing edges from e."""
        # Remove edges that go out of nodes that are in evidence.
        dog_network_test = BayesNet()
        dog_network_test.load_from_bifxml('testing/dog_problem.BIFXML')

        for e in evidence:
            # Get children of the nodes that are in evidence.
            for leaf_node in dog_network_test.get_children(e):
                print(leaf_node)
                # Cut the line between nodes in evidence and its child node.
                dog_network_test.del_edge((e, leaf_node))
        return dog_network_test.draw_structure()

Query = ["light-on"]
evidence = [("family-out"), ("dog-out")]
test = BNReasoner
test.edge_pruning(Query, evidence)
