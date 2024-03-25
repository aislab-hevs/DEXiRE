from typing import Any, Dict, List, Tuple, Union, Callable, Set

from.expression import Expr
from .dexire_abstract import AbstractClause

class ConjuntiveClause(AbstractClause):
  def __init__(self, clausses: List[Union[Expr, AbstractClause]] = []) -> None:
    self.clausses = clausses

  def eval(self, value: Any) -> bool:
    value_list = []
    print(value)
    for i in range(len(self.clausses)):
      value_list.append(self.clausses[i].eval(value[i]))
    return all(value_list)

  def add_clausses(self, clause: List[Expr]):
      self.clausses += clause

  def get_feature_idx(self):
    return [expr.get_feature_idx() for expr in self.clausses]

  def get_feature_name(self):
    return [expr.get_feature_name() for expr in self.clausses]

  def __len__(self):
    return len(self.clausses)

  def __hash__(self) -> int:
    return hash(repr(self))

  def __repr__(self) -> str:
     return self.__str__()

  def __str__(self) -> str:
    if len(self.clausses) == 0:
      return "[]"
    else:
      return "("+" AND ".join([str(c) for c in self.clausses])+")"

  def __eq__(self, other: object) -> bool:
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clausses) == len(other):
        if set(self.clausses) == set(other):
          equality = True
    return equality

  def __iter__(self):
    for expr in self.clausses:
      yield expr


class DisjuntiveClause(AbstractClause):
  def __init__(self, clausses: List[Union[Expr, AbstractClause]] = []) -> None:
    self.clausses = clausses

  def get_feature_idx(self):
    return [expr.get_feature_idx() for expr in self.clausses]

  def get_feature_name(self):
    return [expr.get_feature_name() for expr in self.clausses]

  def add_clausses(self, clause: List[Expr]):
      self.clausses += clause

  def eval(self, value: List[Any]) -> bool:
    value_list = []
    for i in range(len(self.clausses)):
      value_list.append(self.clausses[i].eval(value[i]))
    return any(value_list)

  def __len__(self):
    return len(self.clausses)

  def __repr__(self) -> str:
     return self.__str__()

  def __str__(self) -> str:
    if len(self.clausses) == 0:
      return "[]"
    else:
      return "["+" OR ".join([str(c) for c in self.clausses])+"]"

  def __eq__(self, other: object) -> bool:
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clausses) == len(other):
        if set(self.clausses) == set(other):
          equality = True
    return equality

  def __hash__(self) -> int:
    return hash(repr(self))

  def __iter__(self):
    for expr in self.clausses:
      yield expr