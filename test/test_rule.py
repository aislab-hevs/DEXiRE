# Test one rule with expressions and clauses. 
import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause


@pytest.fixture
def create_rule():
    premise = ConjunctiveClause([Expr(0, 3.0, '>', 'feature_0'), Expr(1, -1.0, '<=', 'feature_1')])
    conclusion = 'Class 1'
    return Rule(premise=premise, 
                conclusion=conclusion, 
                coverage=0.78, 
                accuracy=0.98,
                proba=0.79)

def test_rule(create_rule):
    rule = create_rule
    assert rule.get_feature_idx() == [0, 1]
    assert rule.get_feature_name() == ['feature_0', 'feature_1']
    assert len(rule) == 2
    premise = ConjunctiveClause([Expr(0, 3.0, '>', 'feature_0'), Expr(1, -1.0, '<=', 'feature_1')])
    conclusion = 'Class 1'
    assert rule == Rule(premise=premise, 
                conclusion=conclusion, 
                coverage=0.78, 
                accuracy=0.98,
                proba=0.79)
    
def test_rule_eval(create_rule):
    rule = create_rule
    assert rule.eval([3.1, -1.1]) == 'Class 1'
    assert rule.eval([3.0, -1.1]) is None