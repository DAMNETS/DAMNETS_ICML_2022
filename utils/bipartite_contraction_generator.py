import networkx as nx
import numpy as np


def get_max_degree_bottom_nodes(G, bottom_nodes):
    # Get the top degrees in the right-hand partition of the bipartite graph.
    degrees = G.degree(bottom_nodes)
    max_degree = max(degrees, key=lambda x: x[1])[1]
    # We will concentrate edges onto these nodes.
    max_degree_nodes = [tup[0] for tup in degrees if tup[1] == max_degree]
    return max_degree_nodes


def generate_bipartite_contraction_ts(n, m, p, decay_prop, T, **kwargs):
    G = nx.bipartite.random_graph(n, m, p)
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(G) - top_nodes
    ts = [nx.Graph(G)]
    for t in range(T):
        max_degree_nodes = get_max_degree_bottom_nodes(G, bottom_nodes)[:1]
        decay_nodes = bottom_nodes - set(max_degree_nodes)
        decay_edges = list(G.edges(decay_nodes))
        # Get the non-edges between the max degree nodes, removing the inbetween edges that would break bipartivity.
        non_edges = set(nx.non_edges(G.subgraph(list(top_nodes) + max_degree_nodes)))\
                    - set(nx.non_edges(G.subgraph(list(top_nodes)))) - set(nx.non_edges(G.subgraph(max_degree_nodes)))
        non_edges = list(non_edges)
        decay_n = int(np.ceil(decay_prop * len(non_edges)))
        if len(non_edges) == 0 or len(decay_edges) == 0:
            break
        else:
            remove_edges = np.random.choice(len(decay_edges), size=decay_n, replace=False)
            add_edges = np.random.choice(len(non_edges), size=decay_n, replace=False)
            for ix in remove_edges:
                edge = decay_edges[ix]
                G.remove_edge(edge[0], edge[1])
            for ix in add_edges:
                edge = non_edges[ix]
                G.add_edge(edge[0], edge[1])
            ts.append(nx.Graph(G))
    return ts
