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
        activation_count = matrix[i] / number_of_tokens / 2 ** -(current.depth+1)
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


def add_edges(G, node, pos, x=0, y=0, layer=1):
    if node is not None:
        G.add_node(node.value, pos=(x, y), activation_count=node.activation_count)
        if node.left:
            G.add_edge(node.value, node.left.value)
            pos = add_edges(G, node.left, pos, x - 1 / layer, y - 1, layer + 1)
        if node.right:
            G.add_edge(node.value, node.right.value)
            pos = add_edges(G, node.right, pos, x + 1 / layer, y - 1, layer + 1)
    return pos

def plot_binary_tree(root, matrix, max_activation, file_name="binary_tree_activated.jpg"):
    G = nx.DiGraph()
    pos = add_edges(G, root, pos={})
    
    node_pos = nx.get_node_attributes(G, 'pos')
    activation_counts = nx.get_node_attributes(G, 'activation_count')
    
    plt.figure(figsize=(12, 9))
    
    # Normalize sizes and colors
    sizes = [count * 1000 / max_activation for count in activation_counts.values()]
    colors = list(activation_counts.values())
    
    nx.draw(
        G, 
        node_pos, 
        with_labels=True,
        labels={node: f"{node:.3f}" for node in G.nodes()},
        node_size=sizes, 
        node_color=colors,
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=max_activation,
        font_size=8, 
        font_weight='bold', 
        arrows=False
    )
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_activation)))
    plt.title(f"Binary Tree Visualization (Max Activation: {max_activation:.2f})")
    plt.savefig(file_name, format="jpg", dpi=300, bbox_inches='tight')
    print(f"Binary tree with activation counts saved as {file_name}")
    plt.close()


def fffn2picture(matrix, number_of_tokens, number_of_tree, width_master_node_by_tree, id_matrix):
    matrix = matrix.view(number_of_tree, -1)
    matrix = matrix[:, :-width_master_node_by_tree]
    matrix = matrix.cpu().numpy()
    
    for idx, tree in enumerate(matrix):
        root, max_activation = matrix_to_binary_tree(tree, number_of_tokens)
        if root is not None:
            tree = tree / number_of_tokens
            plot_binary_tree(root, tree, max_activation, f"{id_matrix}_{idx}.jpg")
            print(f"ID: {id_matrix} Tree {idx} done for {number_of_tokens:,} tokens")
        else:
            print(f"ID: {id_matrix} Tree {idx} is empty, skipping visualization")
