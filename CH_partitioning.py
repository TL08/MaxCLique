from strawberryfields.apps import data, plot, sample, clique
import numpy as np
import networkx as nx
import plotly
from __future__ import print_function
import cvxopt.base
import cvxopt.solvers

#Lovasz Number

def parse_graph(G, complement=False):
    '''
    Takes a Sage graph, networkx graph, or adjacency matrix as argument, and returns
    vertex count and edge list for the graph and its complement.
    '''

    if type(G).__module__+'.'+type(G).__name__ == 'networkx.classes.graph.Graph':
        import networkx
        G = networkx.convert_node_labels_to_integers(G)
        nv = len(G)
        edges = [ (i,j) for (i,j) in G.edges() if i != j ]
        c_edges = [ (i,j) for (i,j) in networkx.complement(G).edges() if i != j ]
    else:
        if type(G).__module__+'.'+type(G).__name__ == 'sage.graphs.graph.Graph':
            G = G.adjacency_matrix().numpy()

        G = np.array(G)

        nv = G.shape[0]
        assert len(G.shape) == 2 and G.shape[1] == nv
        assert np.all(G == G.T)

        edges   = [ (j,i) for i in range(nv) for j in range(i) if G[i,j] ]
        c_edges = [ (j,i) for i in range(nv) for j in range(i) if not G[i,j] ]

    for (i,j) in edges:
        assert i < j
    for (i,j) in c_edges:
        assert i < j

    if complement:
        (edges, c_edges) = (c_edges, edges)

    return (nv, edges, c_edges)

def lovasz_theta(G, long_return=False, complement=True):
    '''
    Computes the Lovasz theta number for a graph.
    Takes either a Sage graph or an adjacency matrix as argument.
    If the `long_return` flag is set, returns also the optimal B and Z matrices for the primal
    and dual programs.
    >>> import networkx
    >>> G = networkx.cycle_graph(5)
    >>> abs(np.sqrt(5) - lovasz_theta(G)) < 1e-9
    True
    >>> # Vertices are {0,1}^5, edges between vertices with Hamming distance at most 2.
    >>> H = [[ 1 if bin(i ^ j).count("1") <= 2 else 0 for i in range(32) ] for j in range(32) ]
    >>> abs(16.0/3 - lovasz_theta(H)) < 1e-9
    True
    >>> Hc = np.logical_not(np.array(H))
    >>> abs(6.0 - lovasz_theta(Hc)) < 1e-9
    True
    '''

    (nv, edges, _) = parse_graph(G, complement)
    ne = len(edges)

    # This case needs to be handled specially.
    if nv == 1:
        return 1.0

    c = cvxopt.matrix([0.0]*ne + [1.0])
    G1 = cvxopt.spmatrix(0, [], [], (nv*nv, ne+1))
    for (k, (i, j)) in enumerate(edges):
        G1[i*nv+j, k] = 1
        G1[j*nv+i, k] = 1
    for i in range(nv):
        G1[i*nv+i, ne] = 1

    G1 = -G1
    h1 = -cvxopt.matrix(1.0, (nv, nv))

    sol = cvxopt.solvers.sdp(c, Gs=[G1], hs=[h1])

    if long_return:
        theta = sol['x'][ne]
        Z = np.array(sol['ss'][0])
        B = np.array(sol['zs'][0])
        return { 'theta': theta, 'Z': Z, 'B': B }
    else:
        return sol['x'][ne]

# GBS heurisitc
def st(g):
  len_g = g.number_of_nodes()
  lb = int((len_g)/10)
  ub = int((len_g/5))
  A = (nx.adjacency_matrix(g)).todense()
  postselected = sample.postselect(A, lb , ub)
  samples = sample.to_subgraphs(postselected, g)  # Convert samples into subgraphs
  shrunk = [clique.shrink(s, g) for s in samples]  # Shrink subgraphs to cliques
  searched = [clique.search(s, g, 30) for s in shrunk]  # Perform local search
  clique_sizes = [len(s) for s in searched]
  largest_clique = np.argmax(clique_sizes)  # Identify largest clique found
  return largest_clique

#K-core reduction 
def k_core_red(G, lower):
  G = nx.k_core(G, lower)
  if G.number_of_nodes() > 0:
    v = choice(list(G.nodes()))
    neighbours = list(G.neighbors(v))
    for n in neighbours:
      n_v = list(G.neighbors(v))
      n_n = list(G.neighbors(n))
      common = list(set(n_v) & set(n_n))
      if len(common) < lower - 2:
        G.remove_edge(v, n)
    G = nx.k_core(G, lower)
  return G

#CH partitioning
def DBK_nx(graph,L):
  subgraphs=[]
  solve = []
  k = len(nx.approximation.max_clique(graph)) #Maxclique heuristic, can use GBS heuristic as an alternative
  G = k_core_red(graph,k)
  subgraphs.append(G)
  while (subgraphs != []):
    g = subgraphs.pop()
    min_deg = (g.number_of_nodes())
    for ver in g.nodes:
      if (g.degree[ver] <= min_deg ):
        min_deg = g.degree[ver]
        v = ver
    g1 = g.subgraph(list(g.neighbors(v)))
    g2 = g.copy()
    g2.remove_node(v)
    for g_temp in [g1,g2]:
      g_temp = k_core_red(g_temp,k)
      lb_temp = len(nx.approximation.max_clique(g_temp))
      if lb_temp > k:
        k = lb_temp
      if (g_temp.number_of_nodes() > L):
        subgraphs.append(g_temp)
      else:
        graph_coloring = nx.greedy_color(g_temp) #Chromatic number, can use lovasz number for graphs < 60
        unique_colors = set(graph_coloring.values())
        if (len(unique_colors)>k):
          solve.append(g_temp)
  return solve


