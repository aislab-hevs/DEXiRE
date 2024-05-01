# Test for rule extractor based on decision tree. 
import os
import pytest
import sys
from sklearn.datasets import load_iris, make_classification, make_regression
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.rule_extractors.tree_rule_extractor import TreeRuleExtractor
from dexire.core.rule_set import RuleSet
from dexire.core.dexire_abstract import Mode



@pytest.fixture
def create_classification_binary_dataset():
    X, y = make_classification(n_classes=2, n_features=20, n_samples=100, random_state=42)
    return X, y

@pytest.fixture
def create_iris_dataset():
    X, y = load_iris(return_X_y=True)
    return X, y

@pytest.fixture
def create_regression_dataset():
    X, y = make_regression(n_features=20, n_samples=100, random_state=42)
    return X, y

def test_tree_rule_extractor(create_classification_binary_dataset):
    X, y = create_classification_binary_dataset
    extractor = TreeRuleExtractor(max_depth=5,
                                  mode=Mode.CLASSIFICATION,
                                  class_names=[0, 1])
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
    
def test_tree_rule_extractor_iris(create_iris_dataset):
    X, y = create_iris_dataset
    extractor = TreeRuleExtractor(max_depth=5,
                                  mode=Mode.CLASSIFICATION,
                                  class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
    
def test_tree_rule_extractor_regression(create_regression_dataset):
    X, y = create_regression_dataset
    extractor = TreeRuleExtractor(max_depth=5,
                                  mode=Mode.REGRESSION,
                                  criterion="absolute_error")
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0