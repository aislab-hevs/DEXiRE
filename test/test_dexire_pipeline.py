# Tests for rule extraction full pipeline
import os
import pytest
import sys
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.rule_set import RuleSet
from dexire.dexire import DEXiRE


@pytest.fixture
def create_and_train_model_for_iris_dataset():
    # Load the iris dataset
    X, y = load_iris(return_X_y=True)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Create a simple model 
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    # Train the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50)
    return model, X_train, X_test, y_train, y_test

def test_dexire_tree_rule_extractor(create_and_train_model_for_iris_dataset):
    model, X_train, X_test, y_train, y_test = create_and_train_model_for_iris_dataset
    dexire = DEXiRE(model=model,
                    class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    
    rule_set = dexire.extract_rules(X_train, y_train, layer_idx=-2)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
