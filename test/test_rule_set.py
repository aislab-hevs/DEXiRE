# test for rule set behavior (e.g. rule_set.predict())
import os
import pytest
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule_set import RuleSet
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