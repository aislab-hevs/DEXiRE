import os
import pytest
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.utils.activation_discretizer import digitalize_row, discretize_activation_layer

@pytest.fixture
def create_sample():
    X = np.random.rand(4, 9)
    return X

@pytest.fixture
def fixed_sample():
    X = np.array([
        [-0.12, 0.3, 1.9, 20.1],
        [-45.12, 400.3, 4.9, 0],
        [-1.0, 0.3, 1.0, -2.4]
    ])
    return X

def test_digitalize_row(create_sample):
    X = create_sample
    digitalized_column = digitalize_row(X[0, :], n_bins=2)
    assert digitalized_column.shape[0] == X.shape[1]

def test_vectorization(fixed_sample):
    X = fixed_sample
    digitalized_column = discretize_activation_layer(X, n_bins=2)
    print(digitalized_column)
    assert digitalized_column.shape[0] == X.shape[0]
    assert digitalized_column.shape[1] == X.shape[1]
