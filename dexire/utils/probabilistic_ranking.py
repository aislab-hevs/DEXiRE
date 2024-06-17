from typing import Tuple, List, Union, Dict, Any
import numpy as np
from collections import Counter

def probabilistic_ranking(dict_path_layer: Dict[Any, Any], key: str ='support') -> List[Tuple]:
    """Sort an activation path of values based on a given key.

    :param dict_path_layer: Dictionary containing the neurons to process.
    :type dict_path_layer: Dict
    :param key: Key to sort the values, defaults to 'support'
    :type key: str, optional
    :return: List of sorted values.
    :rtype: List[Tuple]
    """
    sort_out = sorted(dict_path_layer.items(), key=lambda item: item[1][key], reverse=True)
    return sort_out

def entropy(labels:np.array) -> float:
    """Calculate entropy of a list of labels.

    :param labels: Label array.
    :type labels: np.array
    :return: Entropy of label distribution.
    :rtype: float
    """
    n = len(labels)
    counts = np.bincount(labels)
    probabilities = counts[np.nonzero(counts)] / n
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(parent_labels: np.array, children_labels:np.array) -> float:
    """Calculate information gain.

    :param parent_labels: labels before splitting.
    :type parent_labels: np.array
    :param children_labels: List of label splits.
    :type children_labels: np.array
    :return: Information gain for the given split
    :rtype: float
    """
    parent_entropy = entropy(parent_labels)
    children_entropy = sum((len(child) / len(parent_labels)) * entropy(child) for child in children_labels)
    return parent_entropy - children_entropy

def calculate_information_gain_per_feature(X: np.array, y: np.array) -> Dict[int, Dict[Any, Any]]:
    """Calculates the information gain per feature split.

    :param X: matrix features values
    :type X: np.array
    :param y: labels before splitting
    :type y: np.array
    :return: Information gain per split in this feature.
    :rtype: Dict[int, Dict[Any, Any]]
    """
    dict_answer = {}
    # original entropy 
    for feature_idx in range(X.shape[1]):
        unique_values = np.unique(X[:, feature_idx])
        dict_answer[feature_idx] = {}
        for val in unique_values:
            left_indices = X[:, feature_idx] == val
            right_indices = X[:, feature_idx] != val
            y_left = y[left_indices]
            y_right = y[right_indices]
            # calculate information gain
            ig = information_gain(y, [y_left, y_right])
            dict_answer[feature_idx].update({'value': val, 'ig': ig})
    return dict_answer