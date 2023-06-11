# import gzip
# import json

import networkx as nx
import pandas as pd
# from tqdm import tqdm

def load_seventh_grader_network():
    # Read the edge list
    df = pd.read_csv(
        "data/moreno_seventh/out.moreno_seventh_seventh",
        skiprows=2,
        header=None,
        sep=" ",
    )
    df.columns = ["student1", "student2", "count"]

    # Read the node metadata
    meta = pd.read_csv(
        "data/moreno_seventh/ent.moreno_seventh_seventh.student.gender",
        header=None,
    )
    meta.index += 1
    meta.columns = ["gender"]

    # Construct graph from edge list.
    G = nx.DiGraph()
    for row in df.iterrows():
        G.add_edge(row[1]["student1"], row[1]["student2"], count=row[1]["count"])
    # Add node metadata
    for n in G.nodes():
        G.nodes[n]["gender"] = meta.loc[n]["gender"]
    return G


def load_sociopatterns_network():
    # Read the edge list

    df = pd.read_csv(
        "data/out.sociopatterns-infectious",
        sep=" ",
        skiprows=2,
        header=None,
    )
    df = df[[0, 1, 2]]
    df.columns = ["person1", "person2", "weight"]

    G = nx.Graph()
    for row in df.iterrows():
        p1 = row[1]["person1"]
        p2 = row[1]["person2"]
        if G.has_edge(p1, p2):
            G.edges[p1, p2]["weight"] += 1
        else:
            G.add_edge(p1, p2, weight=1)

    for n in sorted(G.nodes()):
        G.nodes[n]["order"] = float(n)

    return G


def load_physicians_network():
    # Read the edge list

    df = pd.read_csv(
        "data/out.moreno_innovation_innovation",
        sep=" ",
        skiprows=2,
        header=None,
    )
    df = df[[0, 1]]
    df.columns = ["doctor1", "doctor2"]

    G = nx.Graph()
    for row in df.iterrows():
        G.add_edge(row[1]["doctor1"], row[1]["doctor2"])

    return G