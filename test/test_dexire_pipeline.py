# Tests for rule extraction full pipeline
import os
import pytest
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.dexire_abstract import Mode, TiebreakerStrategy, RuleExtractorEnum
from dexire.core.rule_set import RuleSet
from dexire.dexire import DEXiRE


@pytest.fixture(scope="module")
def create_and_train_model_for_iris_dataset():
    # Load the iris dataset
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False)
    y_reshaped = np.array(y).reshape(-1, 1)
    y = encoder.fit_transform(y_reshaped)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Create a simple model 
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(y.shape[1], activation='softmax')
    ])
    # Train the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # print(model.summary())
    model.fit(X_train, y_train, epochs=50)
    return model, X_train, X_test, y_train, y_test

def test_rule_extraction_at_layer(create_and_train_model_for_iris_dataset):
    model, X_train, X_test, y_train, y_test = create_and_train_model_for_iris_dataset
    dexire = DEXiRE(model=model,
                    class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
                    mode=Mode.CLASSIFICATION,
                    rule_extraction_method=RuleExtractorEnum.TREERULE
                    )
    rule_set = dexire.extract_rules_at_layer(X_train, y_train, layer_idx=-2)
    print(rule_set)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
    
    
def test_dexire_tree_rule_extractor(create_and_train_model_for_iris_dataset):
    model, X_train, X_test, y_train, y_test = create_and_train_model_for_iris_dataset
    dexire = DEXiRE(model=model,
                    class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    
    rule_set = dexire.extract_rules(X_train, y_train, layer_idx=-2)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
