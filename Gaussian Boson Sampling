pip install strawberryfields --upgrade
from strawberryfields.apps import data, plot, sample, clique
import numpy as np
import networkx as nx
import plotly

TA = data.TaceAs()
A = TA.adj
TA_graph = nx.Graph(A)
plot.graph(TA_graph)
#Creating samples and post-selecting them
smp = sample.sample(A,8,1000,loss=0.1)
postselected = sample.postselect(smp, 6, 10)
print (len(postselected))

#Comparing the densities of the samples
GBS_dens = []
u_dens = []
uni=[]

for s in samples:
    uniform = list(np.random.choice(24, 8, replace=False))
    uni.append(uniform) # generates uniform sample
    GBS_dens.append(nx.density(TA_graph.subgraph(s)))
    u_dens.append(nx.density(TA_graph.subgraph(uniform)))

print("GBS max density = {:.4f}".format(np.max(GBS_dens)))
print("Uniform max density = {:.4f}".format(np.max(u_dens)))

print("GBS mean density = {:.4f}".format(np.mean(GBS_dens)))
print("Uniform mean density = {:.4f}".format(np.mean(u_dens)))

#shrinking to find cliques
shrunk = [clique.shrink(s, TA_graph) for s in samples]  # Shrink subgraphs to cliques
gb_shrunk=[]
gb_slen=[]
for s in shrunk:
  gb_shrunk.append(nx.density(TA_graph.subgraph(s)))
  gb_slen.append(len(s))

# Searching nearby for better cliques
searched = [clique.search(s, TA_graph, 10) for s in shrunk]
clique_sizes = [len(s) for s in searched]
print("First two cliques = ", searched[:2])
print("Average clique size = {:.3f}".format(np.mean(clique_sizes)))
print("Maximum clique size = ", np.max(clique_sizes))
