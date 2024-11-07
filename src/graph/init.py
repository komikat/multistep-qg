import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

with open("../../dataset/MetaQA/kb.txt") as f:
    full = f.read().split("\n")


G = nx.DiGraph()

for line in tqdm(full):
    if len(line.split("|")) == 3:
        movie, rel, ent = line.split("|")
        G.add_node(movie, type="movie")
        G.add_node(ent)
        G.add_edge(movie, ent, relation=rel)

nx.write_graphml(G, "../../dataset/MetaQA/movies.graphml")
print("Graph written to dataset/MetaQA/movies.graphml")
