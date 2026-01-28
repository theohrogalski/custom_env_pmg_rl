import networkx as nx

# Create an undirected graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])

# Get the number of edges for node 1
degree_1 = G.degree(1)
print(f"Node 1 has {degree_1} edges.") # Output: Node 1 has 2 edges.

# Get the number of edges for node 2
degree_2 = G.degree(2)
print(f"Node 2 has {degree_2} edges.") # Output: Node 2 has 3 edges.

# Get degrees for all nodes
print("All degrees:", dict(G.degree()))
# Output: All degrees: {1: 2, 2: 3, 3: 1, 4: 1, 5: 1}[]