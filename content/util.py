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
