import matplotlib.pyplot as plt
from collections import defaultdict
from os.path import splitext
from contextlib import contextmanager
from pathlib import Path
import networkx as nx
from decorator import decorator
from networkx.utils import create_random_state, create_py_random_state
import pandas as pd
import copy
from utils_function import *
from utils_class import *

def get_user(user, dict_users):
    """
	Returns the class User from ID and update the dictionary

            Parameters:
                    user (int): user ID
                    dict_users (dict): as a key we must have the user ID and as a value we must have the class User 

            Returns:
                    user (User): class User
                    dict_users (dict) return the dict_users updated if the user not exists

	"""
	# create user if it wasn't created previously
    if user not in dict_users:
        user_obj = User(user)
        dict_users[user] = user_obj
    return dict_users[user], dict_users


def create_graph(df, graph):
    """
    Returns the updated graph, the first call of this function, the graph must be empty
    
            Parameters:
                    df (dataframe): the dataset
    				type_node(string): the type of dataset (c2a, c2q, a2q)
    				weight(int): the weight to assign for each edge
    				graph(dict): the graph obtained until now, so the first one must be empty
    
            Returns:
                    graph(dict): return the updated graph
    
    """
    # for each row in the dataset
    for index, row in df.iterrows():
        source_ = str(row[0])
        target_ = str(row[1])
        weight = int(row[2])
        source, graph = get_user(source_, graph)
        target, graph = get_user(target_, graph)
		# create Relation(edge) between source(v) and target(u)
        rel = Relation(source, target, weight)
		# add relation in user
        graph[target_].add_in_relation(rel)   # add to u the incoming edge from v
        graph[source_].add_out_relation(rel) #  # add to v the oucoming edge to v   
    return graph

def graph_to_networkx(graph): 
    """
    Trasform the our graph in Networkx object
    
            Parameters:
                    graph (dict): as a key we have the user ID and as a value we have the class associate of the User 
                    type_graph(string): the type of graph that we want to obtain (c2a, c2q, a2q), 
                        NOTE: if type_graph is all, this mean we must to add all type of node
    
            Returns:
                    G(networkx): the graph
    
    """
	
    G = nx.DiGraph()
    for user in graph:
        for target, relation in graph[user].get_out_relation.items():
        #create node
            G.add_nodes_from([user, target])
        # create edge
            G.add_edge(user, target, weight = relation.weight)
    return G

def visualization(graph, source = "A", target = "F", k = 1, seed = 12, cut = []):
    G = graph_to_networkx(graph)
    plt.figure(figsize = (10,8))

    pos = {"A": (0,1),"B": (1,2),"C": (1,0),"D": (2,2),"E": (2,0),"F": (3,1)}
    
    nodes = G.nodes
    
    font = 22
    labels = nx.draw_networkx_labels(G,pos,{i: i for i in nodes}, font_size = font,font_color='black')
    
    nx.draw_networkx_nodes(G, pos, node_size = 2800, nodelist = set(nodes) - set([source,target]), alpha = 1, node_color = "white", label = "neighbour nods", edgecolors = "black")
    nx.draw_networkx_nodes(G, pos, node_size = 2800, nodelist = [source,target],alpha = 0.8, node_color = "gold", label = "start node" ,edgecolors = "darkorange")    
    edge_weight = nx.get_edge_attributes(G,'weight')
    font = 15
    nx.draw_networkx_edges(G,pos, edgelist = G.edges, width = 1, arrowstyle = "-|>", arrowsize = 15, edge_color = "gray", alpha = 1,connectionstyle = "Arc3" , node_size = 5300)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = edge_weight,font_size = font)
    nx.draw_networkx_edges(G,pos, edgelist = cut, width = 1, arrowstyle = "-|>", arrowsize = 15, edge_color = "red", alpha = 1,connectionstyle = "Arc3" , node_size = 5300)
    plt.axis("off")
    plt.show()
    
def createResidualG(graph):
    return copy.deepcopy(graph)

def reachFromS(graph, source):
    """
    Return all node that reach from s
    
            Parameters:
                    graph: the graph we are working on
                    source: the starting node that we would to find all node reach from source
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    result (class): sub graph with the desiderable interval
                    
    """
    path = dict()
    path = BFS(graph, source, 0, path)[1]
    return path.keys()

def getNeighborsMinWeight(node, graph):
    """
    Return a graph with the desiderable interval
    
            Parameters:
                    node: the node that we would to find all neighbor
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    result (dict): contains as key all neighbors and as value for each key the weight 
                    (if exists 2 or more edges between source and neighbor we take the edge with the minimum cost)"
                    
    """
    neighbors = dict()
    x = graph[node].get_out_relation
    for target in x.keys():
            neighbors[target] = min(neighbors.get(target, float('inf')), x[target].weight)
    x = graph[node].get_in_relation
    # we take into account also the the edges in relation
    for source in x.keys():
            neighbors[source] = min(neighbors.get(source, float('inf')), x[source].weight_in)
    return neighbors

def getBottleneck(path, source, target):
    """
    Return the bottleneck, so the edges with the minimum cost in the path
    
            Parameters:
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    bottleneck (int): minimum cost in the path
                    

    """
    current_value = target
    bottleneck = float("inf")
    while current_value != source:
        bottleneck = min(bottleneck, path[current_value][1])
        current_value = path[current_value][0]
    return bottleneck

def BFS(graph, s, t, path, start=200808, end=201603):
    """
    return two values, the first one if exists a path between s and t, and the second one is the path among s and t 
    
            Parameters:
                    graph: the graph we are working on 
                    path(dictionary): as key all node of the graph and as value -1
                    flow(int): value used to update the residual in the path
                    s(int): source -> starting point
                    t(int): target -> end point
            
            Returns:
                    boolean: if exists a path between s and t
                    path(dictionary): as key the node and as value the parent node if exists, otherwise -1

    """
    visited = set()
    queue = [s]
    visited.add(s)
    if s not in path: path[s] = ()
    while len(queue) > 0:
        source = queue.pop(0)
        neighbors = getNeighborsMinWeight(source, graph)
        for target in neighbors:
            if target not in visited and neighbors[target] > 0:
                queue.append(target)
                visited.add(target)
                path[target] = (source, neighbors[target])
    if t in visited:  return True, path       
    return False, path


def updateResGraph(graph, path, flow, s, t):
    """
    Update the residual in the path given in input
    
            Parameters:
                    graph: the graph we are working on 
                    path(dictionary): as key the node and as value the parent node
                    flow(int): value used to update the residual in the path
                    s(int): source -> starting point
                    t(int): target -> end point
    """
    walk = [t]
    current_value = t
    while current_value != s:
        walk.append(path[current_value][0])
        current_value = path[current_value][0]
    walk.reverse() 
    for i in range(len(walk)-1):
        el = walk[i]
        out_ = graph[el].get_out_relation
        for target in out_:
            if out_[target].target == walk[i+1]:
                w = out_[target].weight - flow
                out_[target].set_weight(w)
                w_in = out_[target].weight_in + flow
                out_[target].set_weight_in(w_in)