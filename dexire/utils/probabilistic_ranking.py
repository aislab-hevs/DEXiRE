from typing import Tuple, List, Union, Dict
import numpy as np
from collections import Counter

def probabilistic_ranking(dict_path_layer: Dict, key: str ='support') -> List[Tuple]:
    """Sort an activation path of values based on a given key.

    Args:
        dict_path_layer (Dict): Dictionary for Layer i. 
        key (str, optional): Key to sort the values. Defaults to 'support'.

    Returns:
        List[Tuple]: List of sorted values.
    """
    sort_out = sorted(dict_path_layer.items(), key=lambda item: item[1][key], reverse=True)
    return sort_out

def entropy(labels:np.array) -> float:
    """Calculate entropy of a list of labels."""
    n = len(labels)
    counts = np.bincount(labels)
    probabilities = counts[np.nonzero(counts)] / n
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(parent_labels: np.array, children_labels:np.array):
    """Calculate information gain."""
    parent_entropy = entropy(parent_labels)
    children_entropy = sum((len(child) / len(parent_labels)) * entropy(child) for child in children_labels)
    return parent_entropy - children_entropy