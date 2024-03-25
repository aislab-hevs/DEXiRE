from typing import Any, Dict, List, Tuple, Union, Callable, Set
from enum import Enum
from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np
import sympy as sp
import pandas as pd
import tensorflow as tf
import pickle

class AbstractExpr(ABC):
    """Abstract class for representing
    logical expressions.

    :param ABC: Abstract class definition.
    :type ABC: ABC Abstract class.
    """
    @abstractmethod
    def eval(self, value: Any)->bool:
      """Evaluates the expression.

      :param value: variable current value.
      :type value: Any
      :return: Logical expression evaluation. True if expression evaluates to True, False otherwise.
      :rtype: bool
      """
      pass

class AbstractClause(AbstractExpr):
  """Abstract class for representing
  logical clauses (sets of expressions).

  :param AbstractExpr: Abstract class expression.
  :type AbstractExpr: AbstractExpr class.
  """

  @abstractmethod
  def eval(self, value: Any)->bool:
    """Evaluates the clause.

    :param value: variables current values.
    :type value: Any
    :return: Logical clause evaluation. True if clause evaluates to True, False otherwise.
    """
    pass

class AbstractRule(ABC):
  """Abstract class for representing
  logical rules.

  :param ABC: Abstract class definition.
  :type ABC: ABC Abstract class.
  """

  @abstractmethod
  def eval(value: Any)-> Union[Any, None]:
    pass

class AbstractRuleSet(ABC):
  pass

class AbstractRuleExtractor(ABC):
  @abstractmethod
  def extract_rules(self, X: Any, y: Any)-> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    pass


class TiebreakerStrategy(str, Enum):
  MAJORITY_CLASS = "majority_class"
  MINORITE_CLASS = "minority_class"
  HIGH_PERFORMANCE = "high_performance"
  HIGH_COVERAGE = "high_coverage"
  FIRST_HIT_RULE = "first_hit_rule"

class Operators(str, Enum):
  GREATER_THAN = ">"
  LESS_THAN = "<"
  EQUAL_TO = "=="
  NOT_EQUAL = "!="
  GREATER_OR_EQ = ">="
  LESS_OR_EQ = "<="

class Mode(str, Enum):
  CLASSIFICATION = "classification"
  REGRESSION = "regression"