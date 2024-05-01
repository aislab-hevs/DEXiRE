import os
import pytest
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause

@pytest.fixture
def create_disjunctive_clause():
    return DisjunctiveClause([Expr(0, 1.0, '>', 'feature_0'), Expr(1, -5.0, '<=', 'feature_1')])

@pytest.fixture
def create_conjunctive_clause():
    return ConjunctiveClause([Expr(0, 0.01, '==', 'feature_0'), Expr(1, 450.5690, '<=', 'feature_1')])

def test_disjunctive_clause(create_disjunctive_clause):   
    disjunctive_clause = create_disjunctive_clause
    assert disjunctive_clause.get_feature_idx() == [0, 1]
    assert disjunctive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(disjunctive_clause) == 2
    assert disjunctive_clause == DisjunctiveClause([Expr(0, 1.0, '>', 'feature_0'), Expr(1, -5.0, '<=', 'feature_1')])
    
def test_conjunctive_clause(create_conjunctive_clause):
    conjunctive_clause = create_conjunctive_clause
    assert conjunctive_clause.get_feature_idx() == [0, 1]
    assert conjunctive_clause.get_feature_name() == ['feature_0', 'feature_1']
    assert len(conjunctive_clause) == 2
    assert conjunctive_clause == ConjunctiveClause([Expr(0, 0.01, '==', 'feature_0'), Expr(1, 450.5690, '<=', 'feature_1')])
    
def test_disjunctive_clause_eval(create_disjunctive_clause):
    disjunctive_clause = create_disjunctive_clause
    assert disjunctive_clause.eval([1.01, -4.99]) == True
    assert disjunctive_clause.eval([0.99, -4.0]) == False
    
def test_conjunctive_clause_eval(create_conjunctive_clause):
    conjunctive_clause = create_conjunctive_clause
    assert conjunctive_clause.eval([0.010, 450.568]) == True
    assert conjunctive_clause.eval([0.010, 450.57]) == False
    
def test_disjunction_eval_numpy(create_disjunctive_clause):
    disjunctive_clause = create_disjunctive_clause
    eval_matrix = np.array([
        [1.2, -4.8],
        [0.9, -6.0],
        [0.9, -4.8]])
    expected = np.array([True, True, False])
    prediction = disjunctive_clause.numpy_eval(eval_matrix)
    assert (prediction == expected).all()

def test_conjunction_eval_numpy(create_conjunctive_clause):
    conjunctive_clause = create_conjunctive_clause
    eval_matrix = np.array([
        [0.01, 451.0],
        [0.01, 450.5],
        [0.02, 450.0]])
    expected = np.array([False, True, False])
    prediction = conjunctive_clause.numpy_eval(eval_matrix)
    assert (prediction == expected).all()