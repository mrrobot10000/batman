import numpy as np 
from collections import Counter 
class Node: 
 def __init__(self, attribute=None): 
 self.attribute = attribute 
 self.children = {} 
 self.value = None # For leaf nodes 
def entropy(y): 
 class_counts = Counter(y) 
 entropy_val = 0 
 total_samples = len(y) 
 for count in class_counts.values(): 
 p = count / total_samples 
 entropy_val -= p * np.log2(p) 
 return entropy_val 
def information_gain(X, y, attribute): 
 total_entropy = entropy(y) 
 unique_values = np.unique(X[:, attribute]) 
 weighted_entropy = 0 
 for value in unique_values: 
 subset_indices = np.where(X[:, attribute] == value)[0] 
 subset_entropy = entropy(y[subset_indices]) 
 weight = len(subset_indices) / len(y) 
 weighted_entropy += weight * subset_entropy 
 return total_entropy - weighted_entropy 
def id3(X, y, attributes): 
 if len(set(y)) == 1: 
 leaf = Node() 
 leaf.value = y[0] 
 return leaf 
 if len(attributes) == 0: 
 leaf = Node() 
 leaf.value = Counter(y).most_common(1)[0][0] 
 return leaf 
 gains = [information_gain(X, y, attribute) for attribute in attributes] 
 best_attribute = attributes[np.argmax(gains)] 
 node = Node(best_attribute) 
 unique_values = np.unique(X[:, best_attribute]) 
 for value in unique_values: 
 subset_indices = np.where(X[:, best_attribute] == value)[0] 
 subset_X = X[subset_indices] 
22 | P a g e
 subset_y = y[subset_indices] 
 if len(subset_y) == 0: 
 leaf = Node() 
 leaf.value = Counter(y).most_common(1)[0][0] 
 node.children[value] = leaf 
 else: 
 node.children[value] = id3(subset_X, subset_y, np.setdiff1d(attributes, [best_attribute]))
 return node 
def print_tree(node, depth=0): 
 if node.attribute is None: 
 print(' ' * depth, 'Predict:', node.value) 
 else: 
 print(' ' * depth, 'Attribute', node.attribute) 
 for value, child_node in node.children.items(): 
 print(' ' * (depth + 1), 'Value', value) 
 print_tree(child_node, depth + 2) 
# Example usage: 
X = np.array([ 
 ['Sunny', 'Hot', 'High', 'Weak'], 
 ['Sunny', 'Hot', 'High', 'Strong'], 
 ['Overcast', 'Hot', 'High', 'Weak'], 
 ['Rain', 'Mild', 'High', 'Weak'], 
 ['Rain', 'Cool', 'Normal', 'Weak'], 
 ['Rain', 'Cool', 'Normal', 'Strong'], 
 ['Overcast', 'Cool', 'Normal', 'Strong'], 
 ['Sunny', 'Mild', 'High', 'Weak'], 
 ['Sunny', 'Cool', 'Normal', 'Weak'], 
 ['Rain', 'Mild', 'Normal', 'Weak'], 
 ['Sunny', 'Mild', 'Normal', 'Strong'], 
 ['Overcast', 'Mild', 'High', 'Strong'], 
 ['Overcast', 'Hot', 'Normal', 'Weak'], 
 ['Rain', 'Mild', 'High', 'Strong'] 
]) 
y = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']) 
attributes = [0, 1, 2, 3] # Indices of attributes 
tree = id3(X, y, attributes) 
print_tree(tree)
