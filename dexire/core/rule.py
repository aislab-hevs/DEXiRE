from typing import Any, Callable, Union, List
import numpy as np

from .dexire_abstract import AbstractExpr, AbstractRule

class Rule(AbstractRule):
  """Rule class, that represents a logic rule.

  :param AbstractRule: AbstractRule class, that represents a logic rule.
  :type AbstractRule: AbstractRule
  """
  def __init__(self,
               premise: AbstractExpr,
               conclusion: Any,
               coverage: float = 0.0,
               accuracy: float = 0.0,
               proba: float = 0.0,
               print_stats: bool = False) -> None:
    """Constructor to create a new rule.

    :param premise: The clause expression of the premise for this rule.
    :type premise: AbstractExpr
    :param conclusion: Conclusion of the rule.
    :type conclusion: Any
    :param coverage: coverage percentage, defaults to 0.0
    :type coverage: float, optional
    :param accuracy: rule's accuracy , defaults to 0.0
    :type accuracy: float, optional
    :param proba: rule probability, defaults to 0.0
    :type proba: float, optional
    :param print_stats: boolean value, True to print rule's statistics, defaults to False
    :type print_stats: bool, optional
    """
    self.premise = premise
    self.conclusion = conclusion
    self.activated = False
    self.coverage = coverage
    self.accuracy = accuracy
    self.proba = proba
    self.vec_eval = None
    self.print_stats = print_stats

  def __eval(self, value: Any) -> Any:
    """Evaluate the rule 

    :param value: Rule's inputs
    :type value: Any
    :return: Conclusion if the premise is True, None otherwise.
    :rtype: Any
    """
    if self.premise.eval(value):
      self.activated = True
      return self.conclusion
    else:
      self.activated = False
      return None
    
  def predict(self, X: np.array) -> Any:
    """Predicts a conclusion based on the input features in X.

    :param X: Numpy array that match the number of feature and order in the rule premise.
    :type X: np.array
    :return: Array of conclusions or Nones
    :rtype: Any
    """
    return self.numpy_eval(X)
    
  def numpy_eval(self, X: np.array) -> Any:
    """Eval the rule using a numpy array.

    :param X: numpy array that match the features in the rule's premise.
    :type X: np.array
    :return: Conclusion if the premise is True, None otherwise.
    :rtype: Any
    """
    boolean_prediction = self.premise.numpy_eval(X)
    answer = np.full(boolean_prediction.shape, None)
    answer[boolean_prediction] = self.conclusion
    return answer

  def eval(self, value: Any) -> Any:
    """Eval the rule given a set of feature values.

    :param value: set of features that match rule's inputs.
    :type value: Any
    :return: Conclusion if the premise is satisfied, None otherwise.
    :rtype: Any
    """
    return self.__eval(value)

  def get_feature_idx(self) -> List[int]:
    """Get the feature indexes that compound this rule.

    :return: Return the list of feature index for this rule.
    :rtype: List[int]
    """
    return self.premise.get_feature_idx()

  def get_feature_name(self) -> List[str]:
    """Get the list of features names that compound the rule's premise.

    :return: List of features names that compound the premise of the rule.
    :rtype: List[str]
    """
    return self.premise.get_feature_name()

  def __len__(self) -> int:
    """Return the terms length in the rule.

    :return: Rule's length.
    :rtype: int
    """
    return len(self.premise)

  def __repr__(self) -> str:
    """Rule's string representation.

    :return: Return string representation of the rule.
    :rtype: str
    """
    return self.__str__()

  def __str__(self) -> str:
    """String representation of the rule.

    :return: Rule's string representation.
    :rtype: str
    """
    if self.print_stats:
      return f"(proba: {self.proba} | coverage: {self.coverage}) IF {self.premise} THEN {self.conclusion}"
    return f"IF {self.premise} THEN {self.conclusion}"

  def __eq__(self, other: object) -> bool:
    """Compare two rules.

    :param other: Other rule to compare with. 
    :type other: object
    :return: Boolean indicating  whether the rules are equal (True) or not (False).
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if self.premise == other.premise and self.conclusion == other.conclusion:
        equality = True
    return equality

  def __hash__(self) -> int:
    """Hash function for the rule. 

    :return: hash representation of the rule. 
    :rtype: int
    """
    return hash(repr(self))