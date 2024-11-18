import random

import networkx as nx


def hop_to_highest_degree_node(G):
    node = random.choice(list(G.nodes))
    visited = {node}
    full = []

    while True:
        neighbours = list(G.successors(node)) + list(G.predecessors(node))
        neighbours = [n for n in neighbours if n not in visited]

        if not neighbours or len(visited) > 5:  # restrict to 5 hops
            break

        degrees = [(n, G.degree(n)) for n in neighbours]
        degrees.sort(key=lambda x: x[1], reverse=True)
        new_node = degrees[0][0]

        label = (
            G[node][new_node]["relation"]
            if new_node in G[node]
            else G[new_node][node]["relation"]
        )
        full.append((node, label, new_node))

        visited.add(new_node)
        node = new_node

    return full


if __name__ == "__main__":
    G = nx.read_graphml("../../dataset/MetaQA/movies.graphml")
    print(f"Graph loaded with {len(G.nodes)} nodes!")
    print(hop_to_highest_degree_node(G))
