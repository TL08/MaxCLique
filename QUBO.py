import numpy as np
import networkx as nx
import plotly
from dwave.system import DWaveSampler
from dwave.system import EmbeddingComposite

#Function to check if given graph is a clqiue
def is_clique(G):
    """
  G (networkx.Graph() object)
  checks if clique or not
    """
    n = len(list(G.nodes()))
    m = len(list(G.edges()))
    if m == (n*(n-1))/2:#A complete graph has number of edges equal to n choose 2 for n nodes
        return True
    else:
        return False

#Function returns the Max-Clique QUBO
def maximum_clique_qubo(G):
    """
    parameters: G (networkx.Graph() object), quad_weight (quadratic coefficient)
    description: Finds the maximum clique QUBO for the input graph
    return Q (dictionary) QUBO
    """
    Q = {}
    GC = nx.algorithms.operators.unary.complement(G)#complement of the graph
    A = 1
    B = 2
    for i in list(GC.nodes()):
        Q[(i, i)] = -A
    for a in list(GC.edges()):
        Q[a] = B
    return Q

#Calling D-Wave solvers
G = nx.gnp_random_graph(40, 0.1)
Q=maximum_clique_qubo(G)
sampler = DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi',token="DEV-f3691e87c6238959f4d63bbb897c04eab38e5e92",solver='Advantage_system4.1')
sampler_embedded = EmbeddingComposite(sampler)
response = sampler_embedded.sample_qubo(Q,num_reads = 3000)
for datum in response.data(['sample','energy','num_occurrences']):
