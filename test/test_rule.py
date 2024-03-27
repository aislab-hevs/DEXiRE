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
    return Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])

def test_rule(create_rule):
    rule = create_rule
    assert rule.get_feature_idx() == [0, 1]
    assert rule.get_feature_name() == ['feature_0', 'feature_1']
    assert len(rule) == 2
    assert rule == Rule([Expr(0, 0, '==', 'feature_0'), Expr(0, 0, '==', 'feature_1')])