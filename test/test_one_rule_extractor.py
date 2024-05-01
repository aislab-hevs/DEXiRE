# Tests for rule extraction with One rule method. 
import os
import pytest
import sys
from sklearn.datasets import load_iris, make_classification, load_diabetes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.dexire_abstract import Mode
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.rule_extractors.one_rule_extractor import OneRuleExtractor
from dexire.core.rule_set import RuleSet



@pytest.fixture
def create_classification_binary_dataset():
    X, y = make_classification(n_classes=2, n_features=20, n_samples=100, random_state=42)
    return X, y
@pytest.fixture
def create_iris_dataset():
    X, y = load_iris(return_X_y=True)
    return X, y

@pytest.fixture
def create_diabetes_dataset():
    X, y = load_diabetes(return_X_y=True)
    return X, y

def test_one_rule_extractor(create_classification_binary_dataset):
    X, y = create_classification_binary_dataset
    extractor = OneRuleExtractor(discretize=True)
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
    
def test_one_rule_extractor_iris(create_iris_dataset):
    X, y = create_iris_dataset
    extractor = OneRuleExtractor(discretize=True)
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0
    
def test_one_rule_extractor_regression(create_diabetes_dataset):
    X, y = create_diabetes_dataset
    extractor = OneRuleExtractor(discretize=False,
                                 mode=Mode.REGRESSION)
    rule_set = extractor.extract_rules(X, y)
    assert isinstance(rule_set, RuleSet)
    assert len(rule_set) > 0