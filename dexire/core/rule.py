from typing import Any, Callable, Union

from .dexire_abstract import AbstractExpr, AbstractRule

class Rule(AbstractRule):
  def __init__(self,
               premise: AbstractExpr,
               conclusion: Any,
               coverage: float = 0.0,
               accuracy: float = 0.0,
               proba: float = 0.0,
               print_stats: bool = False) -> None:
    self.premise = premise
    self.conclusion = conclusion
    self.activated = False
    self.coverage = coverage
    self.accuracy = accuracy
    self.proba = proba
    self.vec_eval = None
    self.print_stats = print_stats

  def __eval(self, value: Any) -> Any:
    if self.premise.eval(value):
      self.activated = True
      return self.conclusion
    else:
      self.activated = False
      return None

  def eval(self, value: Any) -> Any:
      return self.__eval(value)

  def get_feature_idx(self):
    return self.premise.get_feature_idx()

  def get_feature_name(self):
    return self.premise.get_feature_name()

  def __len__(self):
    return len(self.clausses)

  def __repr__(self) -> str:
     return self.__str__()

  def __str__(self) -> str:
    if self.print_stats:
      return f"(proba: {self.proba} | coverage: {self.coverage}) IF {self.premise} THEN {self.conclusion}"
    return f"IF {self.premise} THEN {self.conclusion}"

  def __eq__(self, other: object) -> bool:
    equality = False
    if isinstance(other, self.__class__):
      if self.premise == other.premise and self.conclusion == other.conclusion:
        equality = True
    return equality

  def __hash__(self) -> int:
    return hash(repr(self))