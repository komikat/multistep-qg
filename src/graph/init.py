import networkx as nx
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open("../../dataset/MetaQA/kb.txt") as f:
    full = f.read().split("\n")


G = nx.DiGraph()

for line in tqdm(full):
    if len(line.split("|")) == 3:
        movie, rel, ent = line.split("|")
        if not G.has_node(movie):
            embedding = model.encode(movie).tolist()
            G.add_node(movie, type="movie", embedding=json.dumps(embedding))
        if not G.has_node(ent):
            embedding = model.encode(ent).tolist()
            G.add_node(ent, embedding=json.dumps(embedding))
        G.add_edge(movie, ent, relation=rel)

nx.write_graphml(G, "../../dataset/MetaQA/movies.graphml")
print("Graph written to dataset/MetaQA/movies.graphml")
