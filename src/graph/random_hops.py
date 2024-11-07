import networkx as nx
import random

# remember to use init.py to generate graph first!
G = nx.read_graphml("../../dataset/MetaQA/movies.graphml")
print(f"Graph loaded with {len(G.nodes)} nodes!")

def random_hop(G):
    node = random.choice(list(G.nodes))
    neighbours = list (G. successors (node)) + list (G.predecessors (node) )
    hops = random. randint (1, 5)
    for _ in range(min(hops, len(neighbours) )):
        new_node = random. choice (neighbours)
        label = G[node][new_node]["relation"] if new_node in G[node] else G[new_node][node]["relation"]
        neighbours. remove (new_node)
        print(f"{node} - {label} - {new_node}")
        neighbours = list(G.successors (new_node)) + list (G. predecessors (new_node))
        neighbours. remove (node)
        node = new_node
        if not neighbours:
            break

random_hop(G)

