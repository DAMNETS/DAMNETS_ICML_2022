import networkx as nx
import numpy as np


def generate_3_comm_decay_ts(c_sizes, T, p_int=0.7, p_ext=0.01, decay_prop=0.2, **kwargs):
    g = nx.random_partition_graph(c_sizes, p_int, p_ext)
    communities = [g.subgraph(c) for c in g.graph['partition']]
    # decay_edges = list(communities[-1].edges)
    decay_nodes = list(communities[-1].nodes)
    non_decay_nodes = [node for comm in communities[:-1] for node in comm]
    ts = [nx.Graph(g)]
    for t in range(T):
        decay_edges = list(g.edges(decay_nodes))
        # get the non-edges intra-community (remove inter community non-edges)
        non_edges = set(nx.non_edges(g.subgraph(list(decay_nodes) + non_decay_nodes)))\
                    - set(nx.non_edges(g.subgraph(list(non_decay_nodes)))) \
                    - set(nx.non_edges(g.subgraph(list(decay_nodes))))
        non_edges = list(non_edges)
        decay_n = int(np.floor(decay_prop * len(decay_edges)))
        if decay_n == 0:
            break
        else:
            remove_edges = np.random.choice(len(decay_edges), size=decay_n, replace=False)
            add_edges = np.random.choice(len(non_edges), size=decay_n, replace=False)
            for ix in remove_edges:
                edge = decay_edges[ix]
                g.remove_edge(edge[0], edge[1])
            for ix in add_edges:
                edge = non_edges[ix]
                g.add_edge(edge[0], edge[1])
            ts.append(nx.Graph(g))
    return ts

def generate_comm_total_decay_ts(c_sizes, T, p_int=0.7, p_ext=0.01, decay_prop=0.2, **kwargs):
    g = nx.random_partition_graph(c_sizes, p_int, p_ext)
    communities = [g.subgraph(c) for c in g.graph['partition']]
    community_nodes = [[node for node in c] for c in communities]
    # non_decay_nodes = [node for comm in communities[:-1] for node in comm]
    ts = [nx.Graph(g)]
    for t in range(T):
        decay_edges = [e for comm in community_nodes for e in g.subgraph(comm).edges()]
        # inter_community non-edges (don't want to add any of these)
        inter_non_edges = {e for c in community_nodes for e in nx.non_edges(g.subgraph(c))}
        non_edges = set(nx.non_edges(g)) - inter_non_edges
        non_edges = list(non_edges)
        decay_n = int(np.floor(decay_prop * len(decay_edges)))
        if decay_n == 0:
            break
        else:
            remove_edges = np.random.choice(len(decay_edges), size=decay_n, replace=False)
            add_edges = np.random.choice(len(non_edges), size=decay_n, replace=False)
            for ix in remove_edges:
                edge = decay_edges[ix]
                g.remove_edge(edge[0], edge[1])
            for ix in add_edges:
                edge = non_edges[ix]
                g.add_edge(edge[0], edge[1])
            ts.append(nx.Graph(g))
    return ts

if __name__ == '__main__':
    ts = generate_comm_total_decay_ts([10, 10, 10], 10, 0.9, 0.0, 0.9)
