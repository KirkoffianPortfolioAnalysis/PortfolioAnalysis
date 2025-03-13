import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.close('all')
def compute_kirchhoff_index(G):
    """
    For an undirected, connected graph G, compute the Kirchhoff index.
    
    The Kirchhoff index is defined as:
        Kf(G) = n * Trace(L^dagger)
    where L is the Laplacian matrix of G and L^dagger is its Moore–Penrose pseudoinverse.
    
    Returns:
      Kf: Kirchhoff index,
      L: Laplacian matrix (dense numpy array),
      L_pinv: Moore–Penrose pseudoinverse of L.
    """
    n = G.number_of_nodes()
    # Compute the Laplacian matrix (as a dense numpy array)
    L = nx.laplacian_matrix(G).todense()
    # Compute the Moore–Penrose pseudoinverse of L
    L_pinv = np.linalg.pinv(L)
    # Compute the trace of the pseudoinverse
    trace_L_pinv = np.trace(L_pinv)
    # Kirchhoff index
    Kf = n * trace_L_pinv
    return Kf, L, L_pinv

def compute_hitting_times(L_pinv, G):
    """
    Compute the hitting times matrix H for an undirected graph using the formula:
    
        H(i,j) = Vol(G) * (L_pinv[j,j] - L_pinv[i,j])   for i ≠ j,
        H(i,i) = 0.
    
    Here, Vol(G) is the total volume (sum of weighted degrees) of G.
    
    Returns:
      H: an n x n numpy array of hitting times.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    # Compute volume as the sum of weighted degrees.
    vol = sum(dict(G.degree(weight='weight')).values())
    
    # Initialize hitting times matrix
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 0
            else:
                H[i, j] = vol * (L_pinv[j, j] - L_pinv[i, j])
    return H
if __name__ == "__main__":
    # Close any previously open figures
    plt.close('all')
    
    # Create an undirected, weighted graph
    G = nx.Graph()
    nodes = ['A', 'B', 'C', 'D']
    G.add_nodes_from(nodes)
    
    # Define weighted edges (symmetric for undirected graphs)
    edges = [
        ('A', 'B', 2),
        ('B', 'C', 1.5),
       # ('A', 'D', 2.5),
        ('C', 'A', 1.5),
       
       
        
     

    ]
    
    # Add edges with weights to the graph
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    # Compute Kirchhoff index and get Laplacian and its pseudoinverse
    Kf, L, L_pinv = compute_kirchhoff_index(G)
    print("Kirchhoff index Kf(G):", Kf)

    print("\nLaplacian matrix L:")
    print(L)
    print("\nMoore–Penrose inverse of L (L^dagger):")

    print(L_pinv)
    
    # Compute hitting times matrix H
    H = compute_hitting_times(L_pinv, G)
    print("\nHitting times matrix H (rows: source, columns: target):")
    for label, row in zip(nodes, H):
        print(f"{label}: {row}")
    
    # Plot the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
            node_size=1500, font_size=14)
    plt.title(f"Undirected Graph\nKirchhoff index Kf(G) = {Kf:.3f}")

exit()