# Modified NetworkX implementation.

import networkx as nx
from networkx.generators.random_graphs import _random_subset, empty_graph
from networkx.utils import py_random_state


@py_random_state(2)
def barabasi_albert_graph_ts(n, m, seed=None, snapshot_iters=None, **kwargs):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.star_graph(m)
    # Make sure every graph contains all nodes
    G.add_nodes_from([i for i in range(m+1, n)])
    assert len(G) == n
    graph_list = [nx.Graph(G)]
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n-m nodes. The first node is m.
    source = m + 1
    iters = 0
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        source += 1
        iters += 1
        if snapshot_iters is not None:
            if (iters % snapshot_iters == 0 and iters > 0) or iters == n:
                graph_list.append(nx.Graph(G))
        else:
            graph_list.append(nx.Graph(G))
    return graph_list
