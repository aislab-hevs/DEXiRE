from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sympy import Symbol, Eq, Or, And, symbols, lambdify
import sympy as symp
import numpy as np

from .dexire_abstract import AbstractExpr, Operators

class Expr(AbstractExpr):
  """Expression class definition for hold logical expressions.

  :param AbstractExpr: Abstract class for expression.
  :type AbstractExpr: AbstractExpr.
  """
  def __init__(self,
               feature_idx: int,
               threshold: Any,
               operator: Union[str, Callable],
               feature_name: str = ""
               ) -> None:
    """Constructor function for creating a new expression.

    :param feature_idx: feature index within feature matrix. 
    :type feature_idx: int
    :param threshold: Value to compare the expression against.
    :type threshold: Any
    :param operator: Operator to compare the expression against the threshold
    :type operator: Union[str, Callable]
    :param feature_name: feature's name, defaults to ""
    :type feature_name: str, optional
    """
    super(Expr, self).__init__()
    self.feature_idx = feature_idx
    self.threshold = threshold
    self.operator = operator
    self.feature_name = feature_name
    self.symbolic_expression = None
    self.str_template = "({feature} {operator} {threshold})"
    self.vec_eval = None
    self.lambda_func = None
    if len(self.feature_name) == 0:
        self.feature_name = f"feature_{self.feature_idx}"
    self._create_symbolic_expression()

  def __generate_sympy_expr(self) -> symp.Expr:
    """Generates a symbolic expression for this expression.

    :raises e: Error in processing the expression.
    :return: Symbolic expression
    :rtype: symp.Expr
    """
    #Generate the logic expression with sympy.
    try:
      if len(self.feature_name) == 0:
        self.feature_name = f"feature_{self.feature_idx}"
      symbolic_expression = symp.parsing.sympy_parser.parse_expr(self.str_template.format(
        feature=self.feature_name, 
        operator=self.operator, 
        threshold=self.threshold), evaluate=False)
      return symbolic_expression
    except Exception as e:
      print(f"Exception generating symbolic expression: {e}")
      raise e
      
  def _create_symbolic_expression(self) -> None:
    """Create a lambify expression from symbolic expression.
    """
    self.symbolic_expression = self.__generate_sympy_expr()
    print(f"Symbolic expression: {self.symbolic_expression}")
    if self.symbolic_expression is not None:
      symbols_in_expr = list(self.symbolic_expression.free_symbols)
      # lambdify expression
      self.lambda_func = lambdify(symbols_in_expr, self.symbolic_expression, 'numpy')
  
  def numpy_eval(self, X: np.array) -> bool:
    """Eval this expression using a numpy array. 

    :param X: Numpy array with values to replace in the expression. 
    :type X: np.array
    :return: Boolean value indicating if the expression is evaluate to True or False.
    :rtype: bool
    """
    if self.symbolic_expression is None or self.lambda_func is None:
      self._create_symbolic_expression()
    if X.ndim == 1:
      return self.lambda_func(X)
    else:
      return self.lambda_func(X.flatten())
  
  def get_symbolic_expression(self) -> symp.Expr:
    """Returns the symbolic expression of this expression.

    :return: Symbolic expression of this expression.
    :rtype: symp.Expr
    """
    if self.symbolic_expression is None:
      self._create_symbolic_expression()
    return self.symbolic_expression

  def eval(self, value: Any) -> bool:
    """Evaluates the logical expression, returning true or false according to condition.

    :param value: Value for variable to evaluate. 
    :type value: Any
    :raises Exception: Operator not recognized. If the operator in expression is not recognized, an exception will be raised.
    :return: Boolean value given the value in the expression.
    :rtype: bool
    """
    if isinstance(self.operator, Callable):
      return self.operator(value, self.threshold)
    elif self.operator == Operators.GREATER_THAN:
      return value > self.threshold
    elif self.operator == Operators.LESS_THAN:
      return value < self.threshold
    elif self.operator == Operators.EQUAL_TO:
      return value == self.threshold
    elif self.operator == Operators.NOT_EQUAL:
      return value != self.threshold
    elif self.operator == Operators.GREATER_OR_EQ:
      return value >= self.threshold
    elif self.operator == Operators.LESS_OR_EQ:
      return value <= self.threshold
    else:
      raise Exception("Operator not recognized")

  def get_feature_idx(self) -> List[int]:
    """Returns the feature index used in this logical expression.

    :return: numerical index of the feature used in this logical expression.
    :rtype: int
    """
    return [self.feature_idx]

  def get_feature_name(self) -> List[str]:
    """Returns the feature name used in this logical expression.

    :return: name of the feature used in this logical expression.
    :rtype: str
    """
    return [self.feature_name]

  def __len__(self) -> int:
    """Returns the number of features used in this logical expression.

    :return: numbers of the features (atomic terms) used in this logical expression.
    :rtype: int
    """
    return 1

  def __repr__(self) -> str:
    """Returns the representation of the logical expression.

    :return: String representation of the logical expression.
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """Returns the string representation of the logical expression.

    :return: String representation of the logical expression.
    :rtype: str
    """
    if self.feature_name is not None and len(self.feature_name) > 0:
      pass
    else:
      self.feature_name = f"feature_{self.feature_idx}"
    return self.str_template.format(feature=self.feature_name,
                                    operator=self.operator,
                                    threshold=self.threshold)
    
  def get_symbols(self) -> List[symp.Symbol]:
    """Return the symbols employed in the expression.

    :return: List of symbols the expression 
    :rtype: List[symp.Symbol]
    """
    if self.symbolic_expression is None:
      self._create_symbolic_expression()
    return list(self.symbolic_expression.free_symbols)

  def __eq__(self, other: object) -> bool:
    """Compares two logical expressions and return True if they are the same expression, False otherwise.

    :param other: Other logical expression to compare.
    :type other: object
    :return: Boolean value True if the logical expressions are the same, False otherwise.
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if self.feature_idx == other.feature_idx \
       and  self.operator == other.operator and \
       self.threshold == other.threshold:
       equality = True
    return equality

  def __hash__(self) -> int:
    """Returns the hash of the logical expression.

    :return: Hash of the logical expression.
    :rtype: int
    """
    return hash(repr(self))