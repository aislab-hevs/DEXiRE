from typing import Any, Callable, Union, List, Dict
import numpy as np
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             r2_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score, 
                             roc_auc_score)

from .dexire_abstract import AbstractRule, AbstractRuleSet, TiebreakerStrategy, Mode

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
      
      
  def answer_preprocessor(self, 
                 Y_hat: np.array, 
                 tiebreakerStrategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> Any:
    # Check if the tiebreaker strategy is in the tiebreaker enum
    final_answer = []
    decision_path = []
    if tiebreakerStrategy not in TiebreakerStrategy:
      raise ValueError(f"Tiebreaker strategy {tiebreakerStrategy} is not in the tiebreaker enumeration")
    if tiebreakerStrategy == TiebreakerStrategy.MAJORITY_CLASS:
      for i in range(Y_hat.shape[0]):
        mask = Y_hat[i, :] != None
        if np.sum(mask) == 0:
          final_answer.append(self.defaultRule())
          decision_path.append(["default_rule"])
        else:
          classes, counts = np.unique(Y_hat[i, mask], return_counts=True)
          max_class = classes[np.argmax(counts)]
          final_answer.append(max_class)
          rule_mask = Y_hat[i, :] == max_class
          decision_path.append(list(np.array(self.rules)[rule_mask]))
    elif tiebreakerStrategy == TiebreakerStrategy.MINORITE_CLASS:
        for i in range(Y_hat.shape[0]):
          mask = Y_hat[i, :] != None
          if np.sum(mask) == 0:
            final_answer.append(self.defaultRule())
            decision_path.append(["default_rule"])
          else:
            classes, counts = np.unique(Y_hat[i, mask], return_counts=True)
            min_class = classes[np.argmin(counts)]
            final_answer.append(min_class)
            rule_mask = Y_hat[i, :] == min_class
            decision_path.append(list(np.array(self.rules)[rule_mask]))
    elif tiebreakerStrategy == TiebreakerStrategy.HIGH_PERFORMANCE:
        for i in range(Y_hat.shape[0]):
          mask = Y_hat[i, :] != None
          if np.sum(mask) == 0:
            final_answer.append(self.defaultRule())
            decision_path.append(["default_rule"])
          else:
            filtered_rules = list(np.array(self.rules)[mask])
            accuracy = [rule.accuracy for rule in filtered_rules]
            max_accuracy_index = np.argmax(accuracy)
            final_answer.append(filtered_rules[max_accuracy_index].conclusion)
            decision_path.append([filtered_rules[max_accuracy_index]])
    elif tiebreakerStrategy == TiebreakerStrategy.HIGH_COVERAGE:
        for i in range(Y_hat.shape[0]):
          mask = Y_hat[i, :] != None
          if np.sum(mask) == 0:
            final_answer.append(self.defaultRule())
            decision_path.append(["default_rule"])
          else:
            filtered_rules = list(np.array(self.rules)[mask])
            coverage = [rule.coverage for rule in filtered_rules]
            max_coverage_index = np.argmax(coverage)
            final_answer.append(filtered_rules[max_coverage_index].conclusion)
            decision_path.append([filtered_rules[max_coverage_index]])
    elif tiebreakerStrategy == TiebreakerStrategy.FIRST_HIT_RULE:
      for i in range(Y_hat.shape[0]):
        mask = Y_hat[i, :] != None
        if np.sum(mask) == 0:
          final_answer.append(self.defaultRule())
          decision_path.append(["default_rule"])
        else:
          for j in range(Y_hat.shape[1]):
            if Y_hat[i, j]!= None:
              final_answer.append(Y_hat[i, j])
              decision_path.append([self.rules[j]])
              break
    return np.array(final_answer), decision_path
  
  
  def predict_numpy_rules(self, 
                          X: np.array, 
                          tiebreakerStrategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE,
                          return_decision_path: bool = False) -> Any:
    # fast inference using numpy 
    partial_answer = [rule.predict(X) for rule in self.rules]
    Y_hat = np.array(partial_answer)
    final_decision, decision_path = self.answer_preprocessor(Y_hat.T, 
                                                             tiebreakerStrategy)
    if not return_decision_path:
      return final_decision
    else:
      return final_decision, decision_path
      
    

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

  def assess_rule_set(self, 
             X: np.array, 
             y_true: np.array, 
             evaluation_method: Dict[str, Callable] = None, 
             mode: Mode = Mode.CLASSIFICATION) -> Dict[str, float]:
    answer_dict = {}
    if evaluation_method is None:
      if mode == Mode.CLASSIFICATION:
        evaluation_method = {
          "accuracy": accuracy_score,
          "precision": precision_score,
          "recall": recall_score,
          "f1": f1_score,
          "roc_auc": roc_auc_score
        }
      elif mode == Mode.REGRESSION:
        evaluation_method = {
          "mse": mean_squared_error,
          "mae": mean_absolute_error,
          "r2": r2_score
        }
      else:
        raise(f"Mode {mode} not supported")
    for key in evaluation_method.keys():
      y_pred = self.predict_numpy_rules(X)
      answer_dict[key] = evaluation_method[key](y_true, y_pred)