import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
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
    if not matrix:
        return None
    
    # Create the root node with its activation count
    root = TreeNode(matrix[0], 0, 1)
    queue = deque([root])
    i = 1
    max = 0    
    while queue and i < len(matrix):
        current = queue.popleft()
        value = matrix[i] / number_of_tokens
        activation_count = matrix[i] / number_of_tokens / 2 ** -(current.depth+1)
        if activation_count > max:
            max = activation_count
        if i < len(matrix):
            current.left = TreeNode(value, current.depth+1, activation_count)
            queue.append(current.left)
            i += 1
        
        if i < len(matrix):
            current.right = TreeNode(value, current.depth+1, activation_count)
            queue.append(current.right)
            i += 1
    
    return root, max

    

# Function to add edges and node properties (like size and color) to the graph
def add_edges(G, node, pos, sizes, colors, x=0, y=0, layer=1):
    if node is not None:
        # Add node with position and store size/color based on activation count
        G.add_node(node.value, pos=(x, y))
        sizes.append(node.activation_count * 300)  # Size proportional to activation count
        colors.append(node.activation_count)  # Color intensity proportional to activation count
        
        # Recursively add children nodes
        if node.left:
            G.add_edge(node.value, node.left.value)
            pos = add_edges(G, node.left, pos, sizes, colors, x - 1 / layer, y - 1, layer + 1)
        if node.right:
            G.add_edge(node.value, node.right.value)
            pos = add_edges(G, node.right, pos, sizes, colors, x + 1 / layer, y - 1, layer + 1)
    
    return pos

# Function to plot and save the binary tree as a JPEG image
def plot_binary_tree(root, matrix, max, file_name="binary_tree_activated.jpg"):
    G = nx.DiGraph()  # Directed graph for tree structure
    sizes = []  # List to hold node sizes
    colors = []  # List to hold node colors based on activation count
    pos = add_edges(G, root, pos={}, sizes=sizes, colors=colors)
    
    # Extract node positions for visualization
    node_pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(8, 6))
    
    # Normalize the color map based on activation counts
    cmap = plt.cm.Blues
    
    # Draw the nodes with varying sizes and colors
    nx.draw(
        G, 
        node_pos, 
        with_labels=True, 
        #labels={node: f"{node:.3f}" for node in G.nodes()},  # Node labels
        node_size=sizes, 
        node_color=colors, 
        cmap=cmap, 
        vmin=0, vmax=max,  # Normalize color map
        font_size=12, 
        font_weight='bold', 
        arrows=False
    )
    
    # Save the plot as a JPEG file
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max)))  # Color bar
    plt.savefig(file_name, format="jpg")
    print(f"Binary tree with activation counts saved as {file_name}")
    plt.show()


def fffn2picture(matrix : torch.Tensor, number_of_tokens : int, number_of_tree : int, width_master_node_by_tree: int, id_matrix: int):
    matrix = matrix.view(number_of_tree, -1)
    matrix = matrix[:, :-width_master_node_by_tree]
    matrix = matrix.cpu().numpy().tolist()
    for idx, tree in enumerate(matrix):
        root, max = matrix_to_binary_tree(tree, number_of_tokens)
        tree = [t / number_of_tokens for t in tree]        
        plot_binary_tree(root, tree, max, f"{id_matrix}_{idx}.jpg")
        print(f"ID: {id_matrix} Tree {idx} done for {number_of_tokens:,} tokens")