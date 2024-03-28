from typing import Any, Dict, List, Tuple, Union, Callable, Set
import sympy as symp

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
    """_summary_

    :param feature_idx: _description_
    :type feature_idx: int
    :param threshold: _description_
    :type threshold: Any
    :param operator: _description_
    :type operator: Union[str, Callable]
    :param feature_name: _description_, defaults to ""
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

  def __generate_sympy_expr(self):
    #Generate the logic expression with sympy.
    try:
      if self.symbolic_expression is None:
        self.symbolic_expression = symp.parsing.sympy_parser.parse_expr(self.str_template.format(
          feature=self.feature_name, 
          operator=self.operator, 
          threshold=self.threshold), evaluate=False)
    except Exception as e:
      print(f"Error generating symbolic expression: {e}")
      
  def get_symbolic_expression(self) -> symp.Expr:
    if self.symbolic_expression is None:
      self.__generate_sympy_expr()
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

  def get_feature_idx(self) -> int:
    """Returns the feature index used in this logical expression.

    :return: numerical index of the feature used in this logical expression.
    :rtype: int
    """
    return self.feature_idx

  def get_feature_name(self) -> str:
    """Returns the feature name used in this logical expression.

    :return: name of the feature used in this logical expression.
    :rtype: str
    """
    return self.feature_name

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