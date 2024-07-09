from typing import Any, Callable, Union, List, Dict
import numpy as np
import dill
dill.settings['recurse'] = True

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
  """_summary_

  :param AbstractRuleSet: _description_
  :type AbstractRuleSet: _type_
  """
  def __init__(self,
               majority_class: Any = None,
               print_stats = False,
               default_tie_break_strategy: TiebreakerStrategy = TiebreakerStrategy.MAJORITY_CLASS):
    """Constructor method to create a new rule set. 

    :param majority_class: set the majority class in the dataset (only for classification), defaults to None
    :type majority_class: Any, optional
    :param print_stats: Boolean variable to print statistics of the rule set, defaults to False
    :type print_stats: bool, optional
    :param default_tie_break_strategy: Default tie breaker strategy, defaults to TiebreakerStrategy.MAJORITY_CLASS
    :type default_tie_break_strategy: TiebreakerStrategy, optional
    """

    self.rules = []
    self.tie_breaker_strategy = default_tie_break_strategy
    self.majority_class = majority_class
    self.print_stats = print_stats
    
  def get_rules(self) -> List[AbstractRule]:
    """Return the rules associated with this rule set.

    :return: List of rules in the rule set.
    :rtype: List[AbstractRule]
    """
    return self.rules

  def set_print_stats(self, print_stats:bool):
    """Set if statistics are printed or not. 

    :param print_stats: Bool value print statistics if True.
    :type print_stats: bool
    """
    self.print_stats = print_stats

  def defaultRule(self) -> Any:
    """Stablish the default prediction if none rule is activated.

    :return: default prediction.
    :rtype: Any
    """
    return self.majority_class

  def __len__(self) -> int:
    """Returns the number of rules in the rule set.

    :return: The number of rules in the rule set.
    :rtype: int 
    """
    return len(self.rules)

  def add_rules(self, rule: List[AbstractRule]):
    """Add a list of rules to the rule set. 

    :param rule: Rules to be added to this rule set.
    :type rule: List[AbstractRule]
    """
    self.rules += rule
      
      
  def answer_preprocessor(self, 
                 Y_hat: np.array, 
                 tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> Any:
    """Process the predictions to display ordered to the final user.

    :param Y_hat: current predictions.
    :type Y_hat: np.array
    :param tie_breaker_strategy: Strategy to break ties between predictions, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :raises ValueError: Tie breaker strategy is not supported.
    :return: processed predictions.
    :rtype: Any
    """
    final_answer = []
    decision_path = []
    if tie_breaker_strategy not in TiebreakerStrategy:
      raise ValueError(f"Tie breaker strategy {tie_breaker_strategy} is not in the tie breaker enumeration")
    if tie_breaker_strategy == TiebreakerStrategy.MAJORITY_CLASS:
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
    elif tie_breaker_strategy == TiebreakerStrategy.MINORITE_CLASS:
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
    elif tie_breaker_strategy == TiebreakerStrategy.HIGH_PERFORMANCE:
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
    elif tie_breaker_strategy == TiebreakerStrategy.HIGH_COVERAGE:
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
    elif tie_breaker_strategy == TiebreakerStrategy.FIRST_HIT_RULE:
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
                          tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE,
                          return_decision_path: bool = False) -> Any:
    """Generates predictions based on the complete feature numpy array.

    :param X: Complete feature array to be evaluated.
    :type X: np.array
    :param tie_breaker_strategy: Strategy to break ties between rules, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :param return_decision_path: Boolean value to return the decision path lead to decision, defaults to False
    :type return_decision_path: bool, optional
    :return: Set of prediction one per row in the feature matrix X.
    :rtype: Any
    """
    # fast inference using numpy 
    partial_answer = [rule.predict(X) for rule in self.rules]
    Y_hat = np.array(partial_answer)
    final_decision, decision_path = self.answer_preprocessor(Y_hat.T, 
                                                             tie_breaker_strategy)
    if not return_decision_path:
      return final_decision
    else:
      return final_decision, decision_path
      
    

  def __predict_one_row(self, data_row: Any,
                      tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> Any:
    """Predicts a single row of features

    :param data_row: row feature set.
    :type data_row: Any
    :param tie_breaker_strategy: Strategy to break the tie between rules, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :return: Prediction for the given feature row
    :rtype: Any
    """
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
        if tie_breaker_strategy == tie_breaker_strategy.FIRST_HIT_RULE:
          return ans, active_rules
    if tie_breaker_strategy == tie_breaker_strategy.MINORITE_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      min_class = classes[np.argmin(counts)]
      return min_class, active_rules
    elif tie_breaker_strategy == tie_breaker_strategy.HIGH_COVERAGE and len(ans)>0:
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
    elif tie_breaker_strategy == tie_breaker_strategy.MAJORITY_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      max_class = classes[np.argmax(counts)]
      return max_class, active_rules
    elif tie_breaker_strategy == tie_breaker_strategy.HIGH_PERFORMANCE and len(ans)>0:
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

  def predict(self, X: Any, return_decision_path = False, tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> Any:
    """Using the feature input array X predicts the decision on the rule set.

    :param X: Complete feature array.
    :type X: Any
    :param return_decision_path: boolean value to return the rules let to the decision, defaults to False
    :type return_decision_path: bool, optional
    :param tie_breaker_strategy: Strategy to break ties, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :return: Predictions from the rule set. 
    :rtype: Any
    """
    # Prepare the input to predict the ouput
    shape = X.shape
    answers = []
    rules_idx = []
    if len(shape) == 1:
      # is only one row
      ans, active_rules = self.__predict_one_row(X, tie_breaker_strategy=tie_breaker_strategy)
      answers.append(ans)
      rules_idx.append(active_rules)
    elif len(shape) == 2:
      # matrix
      for i in range(X.shape[0]):
        x_row = X[i, :]
        ans, active_rules = self.__predict_one_row(x_row, tie_breaker_strategy=tie_breaker_strategy)
        #print(f"#{ans}")
        answers.append(ans)
        rules_idx.append(active_rules)
    else:
      raise(f"Input cannot be with rank over 2, current rank: {shape}")
    if return_decision_path:
      return answers, rules_idx
    else:
      return answers

  def __str__(self) -> str:
    """Obtain the string representation of the rule set. 

    :return: String representation of the rule set.
    :rtype: str
    """
    for rule in self.rules:
      rule.print_stats = self.print_stats
    return f"{self.rules}"

  def __repr__(self) -> str:
    """Obtain the string representation of the rule set. 

    :return: Rule set string representation.
    :rtype: str
    """
    return self.__str__()

  def assess_rule_set(self, 
             X: np.array, 
             y_true: np.array, 
             evaluation_method: Dict[str, Callable] = None, 
             mode: Mode = Mode.CLASSIFICATION) -> Dict[str, float]:
    """Evaluates the rule set given a numpy array. 

    :param X: Complete feature array. 
    :type X: np.array
    :param y_true: Ground truth values. 
    :type y_true: np.array
    :param evaluation_method: Dictionary of metrics or function to evaluate, defaults to None
    :type evaluation_method: Dict[str, Callable], optional
    :param mode: describes if the evaluation is made for classification or regression, defaults to Mode.CLASSIFICATION
    :type mode: Mode, optional
    :return: Dictionary of metrics results.
    :rtype: Dict[str, float]
    """
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
      
    return answer_dict
  
  def __eq__(self, other: object) -> bool:
      """Compare two rule sets. 

      :param other: Other rule set to compare with.
      :type other: object
      :return: True if the rule sets are equal, False otherwise. 
      :rtype: bool
      """
      equality = False
      if isinstance(other, self.__class__):
        equality = set(self.rules) == set(other.rules)
      return equality
    
  def save(self, filename: str) -> None:
    """Save the current rule set to a binary file with extension (.pkl).

    :param filename: Relative or absolute path to the binary file should end with ".pkl" extension.
    :type filename: str
    """
    with open(filename, mode='wb') as fp:
      dill.dump(self, fp)
      
  def load(self, filename: str) -> None:
    """Load a rule set from a file. 

    :param filename: Relative or absolute file path to the binary file should end with ".pkl" extension.
    :type filename: str
    """
    with open(filename, mode='rb') as fp:
      self = dill.load(fp)
