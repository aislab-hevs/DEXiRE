import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause

@pytest.fixture
def create_disjunctive_clause():
    return DisjunctiveClause([Expr(0, 1.0, '>', 'feature_0'), Expr(1, -5.0, '<=', 'feature_1')])

@pytest.fixture
def create_conjunctive_clause():
    return ConjunctiveClause([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])

def test_disjunctive_clause(create_disjunctive_clause):   
    disjunctive_clause = create_disjunctive_clause
    assert disjunctive_clause.get_feature_idx() == [0, 1]
    assert disjunctive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(disjunctive_clause) == 2
    assert disjunctive_clause == DisjunctiveClause([Expr(0, 0, '==', 'feature_0'), Expr(0, 0, '==', 'feature_1')])
    
def test_conjunctive_clause(create_conjunctive_clause):
    conjunctive_clause = create_conjunctive_clause
    assert conjunctive_clause.get_feature_idx() == [0, 1]
    assert conjunctive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(conjunctive_clause) == 2
    assert conjunctive_clause == ConjunctiveClause([Expr(0, 0, '==', 'feature_0'), Expr(0, 0, '==', 'feature_1')])