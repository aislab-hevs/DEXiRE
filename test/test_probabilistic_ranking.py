# Test one rule with expressions and clauses. 
import os
import pytest
import sys
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dexire.utils.probabilistic_ranking import probabilistic_ranking, entropy, information_gain, calculate_information_gain_per_feature
from dexire.utils.activation_discretizer import discretize_activation_layer

@pytest.fixture
def generate_binary_data():
    X, y = make_classification(random_state=49)
    return X, y

@pytest.fixture
def generate_multi_class_data():
    X, y = load_iris(return_X_y=True)
    return X, y

def test_entropy_on_binary_data(generate_binary_data):
    X, y = generate_binary_data
    predicted = entropy(y)
    assert predicted >= 0 and predicted <= 1.0
    

def test_entropy_on_multi_class_data(generate_multi_class_data):
    X, y = generate_multi_class_data
    predicted = entropy(y)
    assert predicted >= 0

def test_information_gain_on_binary_data(generate_binary_data):
    X, y = generate_binary_data
    # binarize 
    X_binary = discretize_activation_layer(X, n_bins=5)
    val, count  = np.unique(X_binary[:, 0], return_counts=True) 
    split_value = val[np.argmax(count)]
    condition = X_binary[:, 0] == split_value
    y_left = y[condition]
    y_right = y[~condition]
    predicted = information_gain(y, [y_left, y_right])
    assert predicted >= 0 and predicted <= 1.0

def test_information_gain_on_multi_class_data(generate_multi_class_data):
    X, y = generate_multi_class_data
    # binarize 
    X_binary = discretize_activation_layer(X, n_bins=5)
    val, count  = np.unique(X_binary[:, 0], return_counts=True) 
    split_value = val[np.argmax(count)]
    condition = X_binary[:, 1] == split_value
    y_left = y[condition]
    y_right = y[~condition]
    predicted = information_gain(y, [y_left, y_right])
    assert predicted >= 0 and predicted <= 1.0
    
def test_probabilistic_ranking(generate_binary_data):
    X, y = generate_binary_data
    # Binarize 
    X_bin = discretize_activation_layer(X, n_bins=2)
    # Ranking
    dict_values = calculate_information_gain_per_feature(X_bin, y)
    ranking = probabilistic_ranking(dict_values, key='ig')
    #print(ranking)
    assert len(ranking) == X_bin.shape[1]