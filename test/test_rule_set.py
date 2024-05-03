# test for rule set behavior (e.g. rule_set.predict())
import os
import pytest
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule_set import RuleSet, TiebreakerStrategy
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.core.rule import Rule



@pytest.fixture
def create_rule_set_empty():
    return RuleSet()

@pytest.fixture
def create_rule_set_with_some_rules():
    premise = ConjunctiveClause([Expr(0, 3.0, '>', 'feature_0'), Expr(1, -1.0, '<=', 'feature_1')])
    conclusion = 'Class 1'
    rule_1 = Rule(premise=premise, 
                conclusion=conclusion, 
                coverage=0.78, 
                accuracy=0.98,
                proba=0.79)
    rule_set = RuleSet()
    rule_set.add_rules([rule_1])
    return rule_set

@pytest.fixture
def create_rule_set_with_four_rules():
    premise_1 = ConjunctiveClause([Expr(0, 3.0, '>', 'feature_0'), 
                                   Expr(1, -1.0, '<=', 'feature_1')])
    conclusion_1 = 'Class 1'
    premise_2 = ConjunctiveClause([Expr(0, 3.0, '>', 'feature_0'), 
                                   Expr(1, -1.0, '<=', 'feature_1'),
                                   Expr(2, -1.0, '>', 'feature_2')])
    conclusion_2 = 'Class 2'
    premise_3 = ConjunctiveClause([Expr(0, 3.0, '<=', 'feature_0'), 
                                   Expr(1, 10.0, '>', 'feature_1')])
    conclusion_3 = 'Class 3'
    rule_1 = Rule(premise=premise_1, 
                conclusion=conclusion_1, 
                coverage=0.78, 
                accuracy=0.98,
                proba=0.79)
    rule_2 = Rule(premise=premise_2, 
                conclusion=conclusion_2, 
                coverage=0.2, 
                accuracy=0.89,
                proba=0.79)
    rule_3 = Rule(premise=premise_3, 
                conclusion=conclusion_3, 
                coverage=0.50, 
                accuracy=0.70,
                proba=0.79)
    rule_4 = Rule(premise=premise_1, 
                conclusion=conclusion_1, 
                coverage=0.90, 
                accuracy=0.92,
                proba=0.60)
    rule_set = RuleSet(majority_class=conclusion_3)
    rule_set.add_rules([rule_1, rule_2, rule_3, rule_4])
    return rule_set

def test_rule_set(create_rule_set_empty):
    rule_set = create_rule_set_empty
    assert len(rule_set) == 0
    assert rule_set == RuleSet()
    
def test_rule_set_add_rule(create_rule_set_with_some_rules):
    rule_set = create_rule_set_with_some_rules
    assert len(rule_set) == 1
    
def test_rule_set_inference(create_rule_set_with_some_rules):
    rule_set = create_rule_set_with_some_rules
    predictions = rule_set.predict(np.array([[3.1, -2.0], [1, -2.0]]))
    assert predictions[0] == ['Class 1']
    assert predictions[1] == []
    
def test_answer_preprocessor_hit_first(create_rule_set_with_four_rules):
    rule_set = create_rule_set_with_four_rules
    input_values = np.array([
        [None, None, None, None],
        [None, 'Class 2', 'Class 3', None],
        [None, None, 'Class 3', 'Class 1']
        ])
    expected = np.array(['Class 3', 'Class 2', 'Class 3'])
    answer, decision_path = rule_set.answer_preprocessor(input_values, 
                                 tie_breaker_strategy=TiebreakerStrategy.FIRST_HIT_RULE)
    assert (answer == expected).all()
    assert len(decision_path) == len(expected)
    
def test_answer_preprocessor_high_coverage(create_rule_set_with_four_rules):
    rule_set = create_rule_set_with_four_rules
    input_values = np.array([
        [None, None, None, None],
        [None, 'Class 2', 'Class 3', None],
        [None, None, 'Class 3', 'Class 1'],
        ['Class 1', 'Class 2', 'Class 3', 'Class 1']
        ])
    expected = np.array(['Class 3', 'Class 3', 'Class 1', 'Class 1'])
    answer, decision_path = rule_set.answer_preprocessor(input_values, 
                                 tie_breaker_strategy=TiebreakerStrategy.HIGH_COVERAGE)
    assert (answer == expected).all()
    assert len(decision_path) == len(expected)
    
def test_predict_numpy_rules(create_rule_set_with_four_rules):
    rule_set = create_rule_set_with_four_rules
    X = np.array([
        [3.1, -1.0, -0.5],
        [3.1, -0.1, -0.5],
        [2.9, 11.0, -0.5],
    ])
    expected = np.array(['Class 1', 'Class 3', 'Class 3'])
    Y_hat = rule_set.predict_numpy_rules(X,
                                         tie_breaker_strategy=TiebreakerStrategy.FIRST_HIT_RULE)
    print(Y_hat)
    assert (Y_hat == expected).all()
    