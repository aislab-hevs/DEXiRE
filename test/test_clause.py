import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.expression import Expr
from dexire.core.clause import DisjuntiveClause, ConjuntiveClause

@pytest.fixture
def create_disjuntive_clause():
    return DisjuntiveClause([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])

@pytest.fixture
def create_conjuntive_clause():
    return ConjuntiveClause([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])

def test_disjuntive_clause(create_disjuntive_clause):   
    disjuntive_clause = create_disjuntive_clause
    assert disjuntive_clause.get_feature_idx() == [0, 1]
    assert disjuntive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(disjuntive_clause) == 2
    assert disjuntive_clause == DisjuntiveClause([Expr(0, 0, '==', 'feature_0'), Expr(0, 0, '==', 'feature_1')])
    
def test_conjjunetive_clause(create_conjuntive_clause):
    conjuntive_clause = create_conjuntive_clause
    assert conjuntive_clause.get_feature_idx() == [0, 1]
    assert conjuntive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(conjuntive_clause) == 2
    assert conjuntive_clause == ConjuntiveClause([Expr(0, 0, '==', 'feature_0'), Expr(0, 0, '==', 'feature_1')])