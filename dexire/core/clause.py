from typing import Any, Dict, List, Tuple, Union, Callable, Set, Iterator

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
    """Returns the string representation of the conjunctive clause.

    :return: String representation of the conjunctive clause.
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """Returns the string representation of the conjunctive clause.

    :return: String representation of the conjunctive clause.
    :rtype: str
    """
    if len(self.clauses) == 0:
      return "[]"
    else:
      return "("+" AND ".join([str(c) for c in self.clauses])+")"

  def __eq__(self, other: object) -> bool:
    """Compares two conjunctive clauses and return True if they are the same clause, False otherwise.

    :param other: The other conjunctive clause to compare.
    :type other: object
    :return: Boolean value True if the conjunctive clauses are the same, False otherwise.
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clauses) == len(other):
        if set(self.clauses) == set(other):
          equality = True
    return equality

  def __iter__(self) -> Iterator[Union[Expr, AbstractClause]]:
    """Iterates over the expressions in the conjunctive clause.

    :yield: expression in the conjunctive clause.
    :rtype: Iterator[Union[Expr, AbstractClause]]
    """
    for expr in self.clauses:
      yield expr


class DisjunctiveClause(AbstractClause):
  """Disjunctive clause (clauses join with OR).

  :param AbstractClause: Abstract class for clause.
  :type AbstractClause: AbstractClause
  """
  def __init__(self, clauses: List[Union[Expr, AbstractClause]] = []) -> None:
    """Constructor for disjunctive clause. Receives the list of clauses or expressions to join

    :param clauses: List of clauses or expressions to join with OR, defaults to []
    :type clauses: List[Union[Expr, AbstractClause]], optional
    """
    self.clauses = clauses

  def get_feature_idx(self) -> List[int]:
    """Get the feature indexes list used in this disjunctive clause.

    :return: List of feature indexes used in this disjunctive clause.
    :rtype: List[int]
    """
    return [expr.get_feature_idx() for expr in self.clauses]

  def get_feature_name(self) -> List[str]:
    """Get the feature names list used in this disjunctive clause.

    :return: List of feature names used in this disjunctive clause.
    :rtype: List[str]
    """
    return [expr.get_feature_name() for expr in self.clauses]

  def add_clauses(self, clause: List[Expr]) -> None:
    """Add a list of expressions to the disjunctive clause.

    :param clause: List of expressions to add to the disjunctive clause
    :type clause: List[Expr]
    """
    self.clauses += clause

  def eval(self, value: List[Any]) -> bool:
    """Evaluates the disjunctive clause given variable values, returning True if any clause is True,

    :param value: List of values to evaluate the expression.
    :type value: List[Any]
    :return: True if any clause is True, False otherwise.
    :rtype: bool
    """
    value_list = []
    for i in range(len(self.clauses)):
      value_list.append(self.clauses[i].eval(value[i]))
    return any(value_list)

  def __len__(self) -> int:
    """returns the number of features used in this disjunctive clause.

    :return: length of the disjunctive clause.
    :rtype: int
    """
    return len(self.clauses)

  def __repr__(self) -> str:
    """Returns the string representation of the disjunctive clause.

    :return: String representation of the disjunctive clause.
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """Returns string representation of the disjunctive clause.

    :return: String representation of the disjunctive clause.
    :rtype: str
    """
    if len(self.clauses) == 0:
      return "[]"
    else:
      return "["+" OR ".join([str(c) for c in self.clauses])+"]"

  def __eq__(self, other: object) -> bool:
    """Compares two disjunctive clauses and return True if they are the same clause, False otherwise

    :param other: Other disjunctive clause to compare.
    :type other: object
    :return: True if the disjunctive clauses are the same, False otherwise.
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if len(self.clauses) == len(other):
        if set(self.clauses) == set(other):
          equality = True
    return equality

  def __hash__(self) -> int:
    """Returns the hash of the disjunctive clause.

    :return: hashed disjunctive clause.
    :rtype: int
    """
    return hash(repr(self))

  def __iter__(self) -> Iterator[Union[Expr, AbstractClause]]:
    """Iterates over the expressions in the disjunctive clause.

    :yield: expression in the disjunctive clause.
    :rtype: Iterator[Union[Expr, AbstractClause]]
    """
    for expr in self.clauses:
      yield expr