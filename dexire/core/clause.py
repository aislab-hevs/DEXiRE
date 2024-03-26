from typing import Any, Dict, List, Tuple, Union, Callable, Set

from.expression import Expr
from .dexire_abstract import AbstractClause

class ConjunctiveClause(AbstractClause):
  """Create a new conjunctive clause (clause join with AND)

  :param AbstractClause: Abstract class for clause.
  :type AbstractClause:  AbstractClause
  """
  def __init__(self, clauses: List[Union[Expr, AbstractClause]] = []) -> None:
    """Constructor for conjunctive clause. Receives the list of clauses or expressions to join in a disyuntive clause.

    :param clauses: List of clauses or expressions to join, defaults to [].
    :type clauses: List[Union[Expr, AbstractClause]], optional
    """
    self.clauses = clauses

  def eval(self, value: Any) -> bool:
    """Evaluates the conjunctive clause given variable values, returning True if all clauses are True, False otherwise.

    :param value: Values to evaluate the expression. 
    :type value: Any
    :return: Boolean value True if all clauses are True, False otherwise.
    :rtype: bool
    """
    value_list = []
    print(value)
    for i in range(len(self.clauses)):
      value_list.append(self.clauses[i].eval(value[i]))
    return all(value_list)

  def add_clauses(self, clause: List[Expr]) -> None:
    """Add a list of expressions to the conjunctive clause.

    :param clause: List of expressions to add to the conjunctive clause.
    :type clause: List[Expr]
    """
    self.clauses += clause

  def get_feature_idx(self) -> List[int]:
    """Get the feature indexes list used in this conjunctive clause.

    :return: List of feature indexes used in this conjunctive clause.
    :rtype: List[int]
    """
    return [expr.get_feature_idx() for expr in self.clauses]

  def get_feature_name(self) -> List[str]:
    """Get the feature names list used in this conjunctive clause.

    :return: List of feature names used in this conjunctive clause.
    :rtype: List[str]
    """
    return [expr.get_feature_name() for expr in self.clauses]

  def __len__(self) -> int:
    """Get the number of features used in this conjunctive clause.

    :return: Number of features used in this conjunctive clause.
    :rtype: int
    """
    return len(self.clauses)

  def __hash__(self) -> int:
    """Returns the hash of the conjunctive clause.

    :return: Hash of the conjunctive clause.
    :rtype: int
    """
    return hash(repr(self))

  def __repr__(self) -> str:
    """_summary_

    :return: _description_
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """_summary_

    :return: _description_
    :rtype: str
    """
    if len(self.clauses) == 0:
      return "[]"
    else:
      return "("+" AND ".join([str(c) for c in self.clauses])+")"

  def __eq__(self, other: object) -> bool:
    """_summary_

    :param other: _description_
    :type other: object
    :return: _description_
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clauses) == len(other):
        if set(self.clauses) == set(other):
          equality = True
    return equality

  def __iter__(self):
    """_summary_

    :yield: _description_
    :rtype: _type_
    """
    for expr in self.clauses:
      yield expr


class DisjunctiveClause(AbstractClause):
  """_summary_

  :param AbstractClause: _description_
  :type AbstractClause: _type_
  """
  def __init__(self, clauses: List[Union[Expr, AbstractClause]] = []) -> None:
    """_summary_

    :param clauses: _description_, defaults to []
    :type clauses: List[Union[Expr, AbstractClause]], optional
    """
    self.clauses = clauses

  def get_feature_idx(self) -> List[int]:
    """_summary_

    :return: _description_
    :rtype: List[int]
    """
    return [expr.get_feature_idx() for expr in self.clauses]

  def get_feature_name(self) -> List[str]:
    """_summary_

    :return: _description_
    :rtype: List[str]
    """
    return [expr.get_feature_name() for expr in self.clauses]

  def add_clauses(self, clause: List[Expr]) -> None:
    """_summary_

    :param clause: _description_
    :type clause: List[Expr]
    """
    self.clauses += clause

  def eval(self, value: List[Any]) -> bool:
    """_summary_

    :param value: _description_
    :type value: List[Any]
    :return: _description_
    :rtype: bool
    """
    value_list = []
    for i in range(len(self.clauses)):
      value_list.append(self.clauses[i].eval(value[i]))
    return any(value_list)

  def __len__(self) -> int:
    """_summary_

    :return: _description_
    :rtype: int
    """
    return len(self.clauses)

  def __repr__(self) -> str:
    """_summary_

    :return: _description_
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """_summary_

    :return: _description_
    :rtype: str
    """
    if len(self.clauses) == 0:
      return "[]"
    else:
      return "["+" OR ".join([str(c) for c in self.clauses])+"]"

  def __eq__(self, other: object) -> bool:
    """_summary_

    :param other: _description_
    :type other: object
    :return: _description_
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clauses) == len(other):
        if set(self.clauses) == set(other):
          equality = True
    return equality

  def __hash__(self) -> int:
    """_summary_

    :return: _description_
    :rtype: int
    """
    return hash(repr(self))

  def __iter__(self):
    """_summary_

    :yield: _description_
    :rtype: _type_
    """
    for expr in self.clauses:
      yield expr