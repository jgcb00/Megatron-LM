import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import numpy as np
import torch

# Define the TreeNode class to store the activation count
class TreeNode:
    def __init__(self, value, depth, activation_count=1):
        self.value = value
        self.activation_count = activation_count  # Number of times activated
        self.depth = depth
        self.left = None
        self.right = None

# Function to convert matrix into a binary tree (matrix holds activation counts)
def matrix_to_binary_tree(matrix, number_of_tokens):
    if matrix.size == 0:
        return None, 0
    
    # Create the root node with its activation count
    root = TreeNode(matrix[0], 0, 1)
    queue = deque([root])
    i = 1
    max_activation = 0    
    while queue and i < matrix.size:
        current = queue.popleft()
        value = matrix[i] / number_of_tokens
        activation_count = matrix[i] * 2 **(current.depth+1) / number_of_tokens
        if activation_count == np.inf:
            print(f"Activation count is infinite for occ : {matrix[i]}, depth : {current.depth+1}, nb_tokens : {number_of_tokens}, matrix dtype : {matrix.dtype}")
        if activation_count > max_activation:
            max_activation = activation_count
        if i < matrix.size:
            current.left = TreeNode(value, current.depth+1, activation_count)
            queue.append(current.left)
            i += 1
        
        if i < matrix.size:
            current.right = TreeNode(value, current.depth+1, activation_count)
            queue.append(current.right)
            i += 1
    
    return root, max_activation

def plot_binary_tree(root, matrix, max_activation, file_name="binary_tree_activated.png"):
    def add_nodes_edges(node, x, y, level, G):
        if node:
            G.add_node(node, pos=(x, -y), activation=node.activation_count)
            if node.left:
                G.add_edge(node, node.left)
                add_nodes_edges(node.left, x - 1 / (2 ** level), y + 1, level + 1, G)
            if node.right:
                G.add_edge(node, node.right)
                add_nodes_edges(node.right, x + 1 / (2 ** level), y + 1, level + 1, G)

    G = nx.DiGraph()
    add_nodes_edges(root, 0, 0, 1, G)
    
    pos = nx.get_node_attributes(G, 'pos')
    activation_counts = nx.get_node_attributes(G, 'activation')
    
    # Calculate the depth of the tree
    max_depth = max(pos[node][1] for node in pos)
    
    # Create a very large figure
    fig_width = min(32, max(16, len(G) / 250))  # Adjust width based on number of nodes
    fig_height = min(32, max(16, max_depth))  # Adjust height based on tree depth
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    
    # Normalize sizes and colors
    max_size = 100  # Reduced max size due to increased number of nodes
    min_size = 1    # Smaller minimum size
    sizes = [min_size + (count / max_activation) * (max_size - min_size) for count in activation_counts.values()]
    colors = list(activation_counts.values())
    
    nodes = nx.draw_networkx_nodes(
        G, 
        pos,
        node_size=sizes, 
        node_color=colors,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=max_activation,
        ax=ax
    )
    
    # Draw edges with very thin lines
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, width=0.1, alpha=0.5)
    
    # Only label nodes near the top of the tree
    top_nodes = {node: f"{node.value:.3f}" for node in G.nodes() if pos[node][1] > -5}
    nx.draw_networkx_labels(
        G, 
        pos, 
        labels=top_nodes,
        font_size=4, 
        font_weight='bold',
        ax=ax
    )
    
    plt.colorbar(nodes, ax=ax, label='Activation Count')
    ax.set_title(f"Binary Tree Visualization\nNodes: {len(G)}, Max Activation: {max_activation:.2f}", fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(file_name, format="png", dpi=300, bbox_inches='tight')
    print(f"Binary tree with activation counts saved as {file_name}")
    plt.close(fig)
    
def fffn2picture(matrix, number_of_tokens, number_of_tree, width_master_node_by_tree, id_matrix):
    matrix = matrix.view(number_of_tree, -1)
    matrix = matrix[:, :-width_master_node_by_tree]
    print(matrix)
    matrix = matrix.cpu().numpy()
    print(matrix)
    for idx, tree in enumerate(matrix):
        root, max_activation = matrix_to_binary_tree(tree, number_of_tokens)
        if root is not None:
            tree = tree / number_of_tokens
            plot_binary_tree(root, tree, max_activation, f"{id_matrix}_{idx}.png")
            print(f"ID: {id_matrix} Tree {idx} done for {number_of_tokens:,} tokens")
        else:
            print(f"ID: {id_matrix} Tree {idx} is empty, skipping visualization")
