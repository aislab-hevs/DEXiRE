from typing import Any, Callable, Union, List
import numpy as np

from .dexire_abstract import AbstractRule, AbstractRuleSet, TiebreakerStrategy

class RuleSet(AbstractRuleSet):
  def __init__(self,
               majority_class = None,
               print_stats = False,
               defaultTiebreakStrategy: TiebreakerStrategy = TiebreakerStrategy.MAJORITY_CLASS):

    self.rules = []
    self.tiebreakerStrategy = defaultTiebreakStrategy
    self.majority_class = majority_class
    self.print_stats = print_stats

  def set_print_stats(self, print_stats:bool):
    self.print_stats = print_stats

  def defaultRule(self):
    return self.majority_class

  def __len__(self):
    return len(self.rules)

  def add_rules(self, rule: List[AbstractRule]):
      self.rules += rule

  def __predict_one_row(self, data_row,
                      tiebreakerStrategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE):
    ans = []
    active_rules = []
    for idx_rule, rule in enumerate(self.rules):
      col_index = rule.get_feature_idx()
      temp_val = data_row[col_index]
      if temp_val.shape[0] == 1:
        res = rule.eval([temp_val])
      elif temp_val.shape[0] > 1:
        res = rule.eval(temp_val)
      else:
        raise(f"Not elements selected, indexes = {col_index}, data + {data_row}")
      if res:
        #print(f"answer: {res}")
        ans.append(res)
        active_rules.append(idx_rule)
        # check one condition
        if tiebreakerStrategy == tiebreakerStrategy.FIRST_HIT_RULE:
          return ans, active_rules
    if tiebreakerStrategy == tiebreakerStrategy.MINORITE_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      min_class = classes[np.argmin(counts)]
      return min_class, active_rules
    elif tiebreakerStrategy == tiebreakerStrategy.HIGH_COVERAGE and len(ans)>0:
      max_coverage = -2
      best_idx = -1
      for idx, rule in enumerate(active_rules):
        if rule.coverage is not None:
          if rule.coverage > max_coverage:
            max_coverage = rule.coverage
            best_idx = idx
      if best_idx > -1:
        return ans[best_idx], [active_rules[best_idx ]]
      else:
        return [], []
    elif tiebreakerStrategy == tiebreakerStrategy.MAJORITY_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      max_class = classes[np.argmax(counts)]
      return max_class, active_rules
    elif tiebreakerStrategy == tiebreakerStrategy.HIGH_PERFORMANCE and len(ans)>0:
      max_performance = -2
      best_idx = -1
      for idx, rule in enumerate(active_rules):
        if rule.proba is not None:
          if rule.proba > max_performance:
            max_performance = rule.proba
            best_idx = idx
      if best_idx > -1:
        return ans[best_idx], [active_rules[best_idx ]]
      else:
        return [], []
    else:
        return ans, active_rules

  def predict(self, X: Any, return_decision_path = False, tiebreakerStrategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE):
    # Prepare the input to predict the ouput
    shape = X.shape
    answers = []
    rules_idx = []
    if len(shape) == 1:
      # is only one row
      ans, active_rules = self.__predict_one_row(X, tiebreakerStrategy=tiebreakerStrategy)
      answers.append(ans)
      rules_idx.append(active_rules)
    elif len(shape) == 2:
      # matrix
      for i in range(X.shape[0]):
        x_row = X[i, :]
        ans, active_rules = self.__predict_one_row(x_row, tiebreakerStrategy=tiebreakerStrategy)
        #print(f"#{ans}")
        answers.append(ans)
        rules_idx.append(active_rules)
    else:
      raise(f"Input cannot be with rank over 2, current rank: {shape}")
    if return_decision_path:
      return answers, rules_idx
    else:
      return answers

  def __str__(self):
    for rule in self.rules:
      rule.print_stats = self.print_stats
    return f"{self.rules}"

  def __repr__(self) -> str:
    return self.__str__()
  
  def __eq__(self, other: object) -> bool:
    if isinstance(other, RuleSet):
      return self.rules == other.rules
    else:
      return False

  def assess(self, X, y):
    #TODO: implement
    pass