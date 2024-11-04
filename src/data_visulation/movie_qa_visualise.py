import networkx as nx
import matplotlib.pyplot as plt

with open("../data/MetaQA/kb.txt") as f:
    full = f.read().split("\n")


G = nx.DiGraph()

for line in full[:40]:
    movie, rel, ent = line.split("|")
    G.add_node(movie, type="movie")
    G.add_node(ent)
    G.add_edge(movie, ent, relation=rel)

nx.draw(G, with_labels=True)
plt.show()