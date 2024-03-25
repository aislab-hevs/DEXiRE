from typing import Any, Dict, List, Tuple, Union, Callable, Set

from .dexire_abstract import AbstractExpr, Operators

class Expr(AbstractExpr):
  def __init__(self,
               feature_idx: int,
               threshold: Any,
               operator: Union[str, Callable],
               feature_name: str = ""
               ) -> None:
    self.feature_idx = feature_idx
    self.threshold = threshold
    self.operator = operator
    self.feature_name = feature_name
    self.symbolic_expression = None
    self.str_template = "({feature} {operator} {threshold})"
    self.vec_eval = None

  def __generate_sympy_expr(self):
    # to do it generate the expresion
    pass

  def eval(self, value: Any) -> bool:
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

  def get_feature_idx(self):
    return self.feature_idx

  def get_feature_name(self):
    return self.feature_name

  def __len__(self):
    return 1

  def __repr__(self) -> str:
     return self.__str__()

  def __str__(self) -> str:
    if self.feature_name is not None and len(self.feature_name) > 0:
      pass
    else:
      self.feature_name = f"feature_{self.feature_idx}"
    return self.str_template.format(feature=self.feature_name,
                                    operator=self.operator,
                                    threshold=self.threshold)

  def __eq__(self, other: object) -> bool:
    equality = False
    if isinstance(other, self.__class__):
      if self.feature_idx == other.feature_idx \
       and  self.operator == other.operator and \
       self.threshold == other.threshold:
       equality = True
    return equality

  def __hash__(self) -> int:
    return hash(repr(self))