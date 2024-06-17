from typing import Any, Dict, List, Tuple, Union, Callable, Set, Iterator
from sympy import Symbol, Eq, Or, And, symbols, lambdify
import sympy as symp
import numpy as np
from collections import OrderedDict

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
    self.feature_name_to_idx = {}
    self.update_feature_name_to_idx()
    self.symbolic_clause = None
    self.indices = []
    self.lambda_func = None
    self.symbols = []
    self._create_symbolic_clause()
    
  def update_feature_name_to_idx(self):
    """
    Update a dictionary between the features name and indices.
    
    :return: Not return value.
    :rtype: None
    """
    temp_dict = {}
    for clause in self.clauses:
      if isinstance(clause, Expr):
        temp_dict[clause.feature_name] = clause.feature_idx
      else:
        clause.update_feature_name_to_idx()
        temp_dict = {**temp_dict, **clause.feature_name_to_idx}
    sorted_items = sorted(temp_dict.items(), key=lambda x: x[1])
    self.feature_name_to_idx = OrderedDict(sorted_items)
    
  def get_symbols(self):
    """Obtain the symbols in this clause. 

    :return: List of symbols in the clause
    :rtype: List[symp.Symbol]
    """
    symbols = []
    for clause in self.clauses:
      if isinstance(clause, Expr):
        symbols.extend(clause.get_symbols())
      else:
        symbols.extend(clause.get_symbols())
    self.symbols = symbols
    return self.symbols
  
  def _create_symbolic_clause(self) -> None:
    """Transform the clause into a symbolic expression. 
    """
    sym_list = [expr.get_symbolic_expression() for expr in self.clauses]
    self.symbolic_clause = And(*sym_list)
    set_str_symbols = [str(s) for s in self.symbolic_clause.free_symbols]
    symbols_in_expr = []
    self.indices = []
    for symbol_name in self.feature_name_to_idx.keys():
      if symbol_name in set_str_symbols:
        symbols_in_expr.append(Symbol(symbol_name))
        self.indices.append(self.feature_name_to_idx[symbol_name])
    # lambdify expression
    self.lambda_func = lambdify(symbols_in_expr, self.symbolic_clause, 'numpy')
    
  def get_symbolic_clause(self) -> symp.Expr:
    """Get symbolic expression from clause.

    :return: Clause symbolic expression
    :rtype: symp.Expr
    """
    if self.symbolic_clause is None:
      self._create_symbolic_clause()
    return self.symbolic_clause
  
  def numpy_eval(self, X: np.array) -> bool:
    """Eval the clause expression as a numpy expression.

    :param X: Numpy array with the input data.
    :type X: np.array
    :return: Boolean value indicating if the clause is True or False given X. 
    :rtype: bool
    """
    if self.symbolic_clause is None or self.lambda_func is None or len(self.indices)==0:
      self._create_symbolic_clause()
    # filter the matrix 
    if len(self.indices) == 0 or self.lambda_func is None:
      self._create_symbolic_clause()
    if X.ndim == 2:
      x = X[:, self.indices]
      result = np.apply_along_axis(lambda na: self.lambda_func(*na), 1, x)
      return result
    elif X.ndim == 1:
      x = X[self.indices]
      return self.lambda_func(*x)
    else: 
      raise ValueError(f"The input column shape {X.shape[1]} do not coincide with the expected: {len(self.indices)}")(f"The input shape {X.shape} do not coincide with the expected: {len(self.indices)}")
    

  def eval(self, value: List[Any]) -> bool:
    """Evaluates the conjunctive clause given variable values, returning True if all clauses are True, False otherwise.

    :param value: Values to evaluate the expression. 
    :type value: Any
    :return: Boolean value True if all clauses are True, False otherwise.
    :rtype: bool
    """
    value_list = []
    for i in range(len(self.clauses)):
      value_list.append(self.clauses[i].eval(value[i]))
    return all(value_list)

  def add_clauses(self, clause: List[Expr]) -> None:
    """Add a list of expressions to the conjunctive clause.

    :param clause: List of expressions to add to the conjunctive clause.
    :type clause: List[Expr]
    """
    self.clauses += clause
    self.update_feature_name_to_idx()
    self._create_symbolic_clause()

  def get_feature_idx(self) -> List[int]:
    """Get the feature indexes list used in this conjunctive clause.

    :return: List of feature indexes used in this conjunctive clause.
    :rtype: List[int]
    """
    index = []
    for expr in self.clauses:
      index += expr.get_feature_idx()
    return index

  def get_feature_name(self) -> List[str]:
    """Get the feature names list used in this conjunctive clause.

    :return: List of feature names used in this conjunctive clause.
    :rtype: List[str]
    """
    index = []
    for expr in self.clauses:
      index += expr.get_feature_name()
    return index

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
    self.feature_name_to_idx = {}
    self.update_feature_name_to_idx()
    self.symbolic_clause = None
    self.lambda_func = None
    self.indices = []
    self.symbols = []
    self._create_symbolic_clause()
    
  def update_feature_name_to_idx(self):
    """
    Update a dictionary between the features name and indices.
    
    :return: Not return value.
    :rtype: None
    """
    temp_dict = {}
    for clause in self.clauses:
      if isinstance(clause, Expr):
        temp_dict[clause.feature_name] = clause.feature_idx
      else:
        clause.update_feature_name_to_idx()
        temp_dict.update(clause.feature_name_to_idx)
    sorted_items = sorted(temp_dict.items(), key=lambda x: x[1])
    self.feature_name_to_idx = OrderedDict(sorted_items)
    
  def get_symbols(self):
    """Obtain the symbols in this clause. 

    :return: List of symbols in the clause
    :rtype: List[symp.Symbol]
    """
    symbols = []
    for clause in self.clauses:
      if isinstance(clause, Expr):
        symbols.extend(clause.get_symbols())
      else:
        symbols.extend(clause.get_symbols())
    self.symbols = symbols
    return self.symbols
    
  def _create_symbolic_clause(self) -> None:
    """Generate the symbolic expression for this clause.
    """
    sym_list = [expr.get_symbolic_expression() for expr in self.clauses]
    self.symbolic_clause = Or(*sym_list)
    set_str_symbols = [str(s) for s in self.symbolic_clause.free_symbols]
    symbols_in_expr = []
    self.indices = []
    if len(self.feature_name_to_idx) == 0:
      self.update_feature_name_to_idx()
    for symbol_name in self.feature_name_to_idx.keys():
      if symbol_name in set_str_symbols:
        symbols_in_expr.append(Symbol(symbol_name))
        self.indices.append(self.feature_name_to_idx[symbol_name])
    # lambdify expression
    self.lambda_func = lambdify(symbols_in_expr, self.symbolic_clause, 'numpy')
    
  def get_symbolic_clause(self) -> symp.Expr:
    """Return symbolic expression for this clause.

    :return: SymPY expression
    :rtype: sympy.Expr
    """
    if self.symbolic_clause is None:
      self._create_symbolic_clause()
    return self.symbolic_clause

  def get_feature_idx(self) -> List[int]:
    """Get the feature indexes list used in this disjunctive clause.

    :return: List of feature indexes used in this disjunctive clause.
    :rtype: List[int]
    """
    index = []
    for expr in self.clauses:
      index += expr.get_feature_idx()
    return index

  def get_feature_name(self) -> List[str]:
    """Get the feature names list used in this disjunctive clause.

    :return: List of feature names used in this disjunctive clause.
    :rtype: List[str]
    """
    index = []
    for expr in self.clauses:
      index += expr.get_feature_name()
    return index

  def add_clauses(self, clause: List[Expr]) -> None:
    """Add a list of expressions to the disjunctive clause.

    :param clause: List of expressions to add to the disjunctive clause
    :type clause: List[Expr]
    """
    self.clauses += clause
    self.update_feature_name_to_idx()
    self._create_symbolic_clause()

  def numpy_eval(self, X: np.array) -> bool:
    """Eval the expression using the array X.

    :param X: Input data for the clause.
    :type X: np.array
    :raises ValueError: The number of features in X does not match the expect input data.
    :return: Boolean expression indicating if the clause is True or False given X.
    :rtype: bool
    """
    if self.symbolic_clause is None or self.lambda_func is None or len(self.indices) == 0:
      self._create_symbolic_clause()
    if X.ndim == 2:
      x = X[:, self.indices]
      result = np.apply_along_axis(lambda na: self.lambda_func(*na), 1, x)
      return result
    elif X.ndim == 1:
      x = X[self.indices]
      return self.lambda_func(*x)
    else: 
      raise ValueError(f"The input column shape {X.shape[1]} do not coincide with the expected: {len(self.indices)}")(f"The input shape {X.shape} do not coincide with the expected: {len(self.indices)}")
    

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