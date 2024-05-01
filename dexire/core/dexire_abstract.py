from typing import Any, List, Union, Set
from enum import Enum
from abc import ABC, abstractmethod

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
    """Evaluates the rule returning the conclusion if the rule evaluates to True. 
    Return None if the rule evaluates to False.

    :param value: variables current values.
    :type value: Any
    :return: Logical rule evaluation. Conclusion if rule evaluates to True, None otherwise.
    """
    pass

class AbstractRuleSet(ABC):
  """Rule set abstract class definition.

  :param ABC: Abstract class base definition.
  :type ABC: ABC abstract class. 
  """
  pass

class AbstractRuleExtractor(ABC):
  """Abstract class for extracting rule sets from data and models.

  :param ABC: Abstract class base definition.
  :type ABC: ABC abstract class.
  """
  @abstractmethod
  def extract_rules(self, X: Any, y: Any)-> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Extracts rule sets from data and models (classification and regression).

    :param X: Input data (features).
    :type X: Any
    :param y: Input labels (predicted labels).
    :type y: Any
    """
    pass


class TiebreakerStrategy(str, Enum):
  """Enumeration strategy for tiebreaker, to break ties in the rule set.

  :param str: string class definition.
  :type str: str.
  :param Enum: enumeration class definition.
  :type Enum: Enum class.
  """
  MAJORITY_CLASS = "majority_class"
  MINORITE_CLASS = "minority_class"
  HIGH_PERFORMANCE = "high_performance"
  HIGH_COVERAGE = "high_coverage"
  FIRST_HIT_RULE = "first_hit_rule"

class Operators(str, Enum):
  """Enumeration operators for logical expressions.

  :param str: String class definition.
  :type str: str.
  :param Enum: Enumeration class definition.
  :type Enum: Enum class.
  """
  GREATER_THAN = ">"
  LESS_THAN = "<"
  EQUAL_TO = "=="
  NOT_EQUAL = "!="
  GREATER_OR_EQ = ">="
  LESS_OR_EQ = "<="

class Mode(str, Enum):
  """Task extraction mode classification or regression.

  :param str: String class definition.
  :type str: str
  :param Enum: Enumeration class definition.
  :type Enum: Enum class.
  """
  CLASSIFICATION = "classification"
  REGRESSION = "regression"
  
class RuleExtractorEnum(str, Enum):
  """Enumerates the different rule extraction approaches

  :param str: String class definition.
  :type str: str
  :param Enum: Enumeration class definition.
  :type Enum: Enum class.
  """
  
  ONERULE = "oneR"
  TREERULE = "treeR"
  MIXED = "mixed" # Combines one rule and tree rule extractor 