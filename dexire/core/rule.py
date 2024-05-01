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
    """_summary_

    :param premise: _description_
    :type premise: AbstractExpr
    :param conclusion: _description_
    :type conclusion: Any
    :param coverage: _description_, defaults to 0.0
    :type coverage: float, optional
    :param accuracy: _description_, defaults to 0.0
    :type accuracy: float, optional
    :param proba: _description_, defaults to 0.0
    :type proba: float, optional
    :param print_stats: _description_, defaults to False
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
    """_summary_

    :param value: _description_
    :type value: Any
    :return: _description_
    :rtype: Any
    """
    if self.premise.eval(value):
      self.activated = True
      return self.conclusion
    else:
      self.activated = False
      return None
    
  def predict(self, X: np.array) -> Any:
    index_list = self.get_feature_idx()
    if X.ndim == 1:
      # check if match the premise symbols
      if X.shape[0] == len(index_list):
        return self.numpy_eval(X)
      elif X.shape[0] > len(index_list):
        return self.numpy_eval(X[index_list])
      else: 
        raise(f"The input shape {X.shape} do not coincide with the expected: {len(index_list)}")
    elif X.ndim == 2:
      #check if columns comply with symbols if more filter the columns
      if X.shape[1] == len(index_list):
        return self.numpy_eval(X)
      elif X.shape[1] > len(index_list):
        return self.numpy_eval(X[:, index_list])
      else:
        raise(f"The input column shape {X.shape[1]} do not coincide with the expected: {len(index_list)}")
    else:
      raise(f"Input cannot be with rank over 2, current rank: {X.dim}")
    
  def numpy_eval(self, X: np.array) -> Any:
    boolean_prediction = self.premise.numpy_eval(X)
    answer = np.full(boolean_prediction.shape, None)
    answer[boolean_prediction] = self.conclusion
    return answer

  def eval(self, value: Any) -> Any:
    """_summary_

    :param value: _description_
    :type value: Any
    :return: _description_
    :rtype: Any
    """
    return self.__eval(value)

  def get_feature_idx(self) -> List[int]:
    """_summary_

    :return: _description_
    :rtype: List[int]
    """
    return self.premise.get_feature_idx()

  def get_feature_name(self) -> List[str]:
    """_summary_

    :return: _description_
    :rtype: List[str]
    """
    return self.premise.get_feature_name()

  def __len__(self) -> int:
    """_summary_

    :return: _description_
    :rtype: int
    """
    return len(self.premise)

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
    if self.print_stats:
      return f"(proba: {self.proba} | coverage: {self.coverage}) IF {self.premise} THEN {self.conclusion}"
    return f"IF {self.premise} THEN {self.conclusion}"

  def __eq__(self, other: object) -> bool:
    """_summary_

    :param other: _description_
    :type other: object
    :return: _description_
    :rtype: bool
    """
    equality = False
    if isinstance(other, self.__class__):
      if self.premise == other.premise and self.conclusion == other.conclusion:
        equality = True
    return equality

  def __hash__(self) -> int:
    """_summary_

    :return: _description_
    :rtype: int
    """
    return hash(repr(self))