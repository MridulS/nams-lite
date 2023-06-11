"""
Solutions to Intro Chapter.
"""


def node_metadata(G):
    """Counts of students of each gender."""
    from collections import Counter

    mf_counts = Counter([d["gender"] for n, d in G.nodes(data=True)])
    return mf_counts


def edge_metadata(G):
    """Maximum number of times that a student rated another student."""
    counts = [d["count"] for n1, n2, d in G.edges(data=True)]
    maxcount = max(counts)
    return maxcount


def adding_students(G):
    """How to nodes and edges to a graph."""
    G = G.copy()
    G.add_node(30, gender="male")
    G.add_node(31, gender="female")
    G.add_edge(30, 31, count=3)
    G.add_edge(31, 30, count=3)  # reverse is optional in undirected network
    G.add_edge(30, 7, count=3)  # but this network is directed
    G.add_edge(7, 30, count=3)
    G.add_edge(31, 7, count=3)
    G.add_edge(7, 31, count=3)
    return G


def unrequitted_friendships_v1(G):
    """Answer to unrequitted friendships problem."""
    unrequitted_friendships = []
    for n1, n2 in G.edges():
        if not G.has_edge(n2, n1):
            unrequitted_friendships.append((n1, n2))
    return unrequitted_friendships


def unrequitted_friendships_v2(G):
    """Alternative answer to unrequitted friendships problem. By @schwanne."""
    return len([(n1, n2) for n1, n2 in G.edges() if not G.has_edge(n2, n1)])


def unrequitted_friendships_v3(G):
    """Alternative answer to unrequitted friendships problem. By @end0."""
    links = ((n1, n2) for n1, n2, d in G.edges(data=True))
    reverse_links = ((n2, n1) for n1, n2, d in G.edges(data=True))

    return len(list(set(links) - set(reverse_links)))



"""Solutions to Hubs chapter."""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import nxviz as nv
from nxviz import annotate

def ecdf(data):
    return np.sort(data), np.arange(1, len(data) + 1) / len(data)

def rank_ordered_neighbors(G):
    """
    Uses a pandas Series to help with sorting.
    """
    s = pd.Series({n: len(list(G.neighbors(n))) for n in G.nodes()})
    return s.sort_values(ascending=False)


def rank_ordered_neighbors_original(G):
    """Original implementation of rank-ordered number of neighbors."""
    return sorted(G.nodes(), key=lambda x: len(list(G.neighbors(x))), reverse=True)


def rank_ordered_neighbors_generator(G):
    """
    Rank-ordered generator of neighbors.

    Contributed by @dgerlanc.

    Ref: https://github.com/ericmjl/Network-Analysis-Made-Simple/issues/75
    """
    gen = ((len(list(G.neighbors(x))), x) for x in G.nodes())
    return sorted(gen, reverse=True)


def ecdf_degree_centrality(G):
    """ECDF of degree centrality."""
    x, y = ecdf(list(nx.degree_centrality(G).values()))
    plt.scatter(x, y)
    plt.xlabel("degree centrality")
    plt.ylabel("cumulative fraction")


def ecdf_degree(G):
    """ECDF of degree."""
    num_neighbors = [len(list(G.neighbors(n))) for n in G.nodes()]
    x, y = ecdf(num_neighbors)
    plt.scatter(x, y)
    plt.xlabel("degree")
    plt.ylabel("cumulative fraction")


def num_possible_neighbors():
    """Answer to the number of possible neighbors for a node."""
    return r"""
The number of possible neighbors can either be defined as:

1. All other nodes but myself
2. All other nodes and myself

If $K$ is the number of nodes in the graph,
then if defined as (1), $N$ (the denominator) is $K - 1$.
If defined as (2), $N$ is equal to $K$.
"""


def circos_plot(G):
    """Draw a Circos Plot of the graph."""
    # c = CircosPlot(G, node_order="order", node_color="order")
    # c.draw()
    nv.circos(G, sort_by="order", node_color_by="order")
    annotate.node_colormapping(G, color_by="order")


def visual_insights():
    """Visual insights from the Circos Plot."""
    return """
We see that most edges are "local" with nodes
that are proximal in order.
The nodes that are weird are the ones that have connections
with individuals much later than itself,
crossing larger jumps in order/time.

Additionally, if you recall the ranked list of degree centralities,
it appears that these nodes that have the highest degree centrality scores
are also the ones that have edges that cross the circos plot.
"""


def dc_node_order(G):
    """Comparison of degree centrality by maximum difference in node order."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import networkx as nx

    # Degree centralities
    dcs = pd.Series(nx.degree_centrality(G))

    # Maximum node order difference
    maxdiffs = dict()
    for n, d in G.nodes(data=True):
        diffs = []
        for nbr in G.neighbors(n):
            diffs.append(abs(G.nodes[nbr]["order"] - d["order"]))
        maxdiffs[n] = max(diffs)
    maxdiffs = pd.Series(maxdiffs)

    ax = pd.DataFrame(dict(degree_centrality=dcs, max_diff=maxdiffs)).plot(
        x="degree_centrality", y="max_diff", kind="scatter"
    )
