import os
import pytest
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.expression import Expr

@pytest.fixture
def create_expression():
    return Expr(0, 0.0, '==', 'feature_0')

@pytest.fixture
def create_expression_less():
    return Expr(0, 0.13890, '<', 'feature_0')

@pytest.fixture
def create_expression_less_or_eq():
    return Expr(0, 2.678900, '<=', 'feature_0')

@pytest.fixture
def create_expression_greater():
    return Expr(0, -10.56789, '>', 'feature_0')

@pytest.fixture
def create_expression_greater_or_eq():
    return Expr(0, 189.0008765434, '>=', 'feature_0')

def test_expression(create_expression):
    expr = create_expression
    assert expr.get_feature_idx() == [0]
    assert expr.get_feature_name() == ['feature_0']
    assert len(expr) == 1
    assert expr == Expr(0, 0, '==', 'feature_0')
    
def test_expression_eq_eval(create_expression):
    expr = create_expression
    assert expr.eval(0) == True
    assert expr.eval(1) == False
    
def test_expression_less_eval(create_expression_less):
    expr = create_expression_less
    assert expr.eval(0.13890) == False
    assert expr.eval(0.13889) == True
    
def test_expression_less_or_eq_eval(create_expression_less_or_eq):
    expr = create_expression_less_or_eq
    assert expr.eval(2.678899) == True
    assert expr.eval(2.678900) == True
    assert expr.eval(2.678901) == False
    
def test_expression_greater_eval(create_expression_greater):
    expr = create_expression_greater
    assert expr.eval(-10.56789) == False
    assert expr.eval(-10.56787) == True
    
def test_expression_greater_or_eq_eval(create_expression_greater_or_eq):
    expr = create_expression_greater_or_eq
    assert expr.eval(189.0008765433) == False
    assert expr.eval(189.0008765434) == True
    assert expr.eval(189.0008765435) == True
    
def test_symbolic_expression_generation_eq(create_expression):
    expr = create_expression
    symbolic_expr = expr.get_symbolic_expression()
    assert symbolic_expr is not None

def test_symbolic_expression_generation_less(create_expression_less):
    expr = create_expression_less
    symbolic_expr = expr.get_symbolic_expression()
    assert symbolic_expr is not None
    
def test_symbolic_expression_generation_less_or_eq(create_expression_less_or_eq):
    expr = create_expression_less_or_eq
    symbolic_expr = expr.get_symbolic_expression()
    assert symbolic_expr is not None
    
def test_symbolic_expression_generation_greater(create_expression_greater):
    expr = create_expression_greater
    symbolic_expr = expr.get_symbolic_expression()
    assert symbolic_expr is not None
    
def test_symbolic_expression_generation_greater_or_eq(create_expression_greater_or_eq):
    expr = create_expression_greater_or_eq
    symbolic_expr = expr.get_symbolic_expression()
    assert symbolic_expr is not None
    
def test_numpy_eval_equal(create_expression):
    expr = create_expression
    input_values = np.array([0.0, 1.0, -1.0])
    expected = np.array([True, False, False])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()
    
def test_numpy_eval_less_or_eq(create_expression_less_or_eq):
    expr = create_expression_less_or_eq
    input_values = np.array([2.678900, 2.0, 2.7])
    expected = np.array([True, True, False])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()
    
def test_numpy_eval_less_eval(create_expression_less):
    expr = create_expression_less
    input_values = np.array([0.13890, 0.13889, 0.13888])
    expected = np.array([False, True, True])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()
    
def test_numpy_eval_greater_eval(create_expression_greater):
    expr = create_expression_greater
    input_values = np.array([-10.56789, -10.56787, -10.56786])
    expected = np.array([False, True, True])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()
    
def test_numpy_eval_greater_or_eq_eval(create_expression_greater_or_eq):
    expr = create_expression_greater_or_eq
    input_values = np.array([189.0008765433, 189.0008765434, 189.0008765435])
    expected = np.array([False, True, True])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()
    
def test_numpy_eval_with_different_shape(create_expression_greater_or_eq):
    expr = create_expression_greater_or_eq
    input_values = np.array([[189.0008765433], [189.0008765434], [189.0008765435]])
    expected = np.array([False, True, True])
    result = expr.numpy_eval(input_values)
    assert (result == expected).all()