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
                 Y_hat: np.ndarray, 
                 activation_mask: np.ndarray = None,
                 tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> Any:
    """Process the predictions to display ordered to the final user.

    :param Y_hat: current predictions.
    :type Y_hat: np.ndarray
    :param tie_breaker_strategy: Strategy to break ties between predictions, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :raises ValueError: Tie breaker strategy is not supported.
    :return: processed predictions.
    :rtype: Any
    """
    final_answer = []
    decision_path = []
    rules_array = np.array(self.rules, dtype=object)
    if not isinstance(tie_breaker_strategy, TiebreakerStrategy):
      raise ValueError(f"Tie breaker strategy {tie_breaker_strategy} is not in the tie breaker enumeration")
    if activation_mask is None:
      activation_mask = Y_hat != None
    for i in range(Y_hat.shape[0]):
      row_mask = activation_mask[i, :]
      if np.sum(row_mask) == 0:
        final_answer.append(self.defaultRule())
        decision_path.append(["default_rule"])
        continue
      active_predictions = Y_hat[i, row_mask]
      active_rules = list(rules_array[row_mask])
      if tie_breaker_strategy == TiebreakerStrategy.MAJORITY_CLASS:
        classes, counts = np.unique(active_predictions, return_counts=True)
        selected = classes[np.argmax(counts)]
        final_answer.append(selected)
        rule_mask = row_mask & (Y_hat[i, :] == selected)
        decision_path.append(list(rules_array[rule_mask]))
      elif tie_breaker_strategy == TiebreakerStrategy.MINORITE_CLASS:
        classes, counts = np.unique(active_predictions, return_counts=True)
        selected = classes[np.argmin(counts)]
        final_answer.append(selected)
        rule_mask = row_mask & (Y_hat[i, :] == selected)
        decision_path.append(list(rules_array[rule_mask]))
      elif tie_breaker_strategy == TiebreakerStrategy.HIGH_PERFORMANCE:
        accuracies = [rule.accuracy for rule in active_rules]
        best_idx = int(np.argmax(accuracies))
        final_answer.append(active_rules[best_idx].conclusion)
        decision_path.append([active_rules[best_idx]])
      elif tie_breaker_strategy == TiebreakerStrategy.HIGH_COVERAGE:
        coverages = [rule.coverage for rule in active_rules]
        best_idx = int(np.argmax(coverages))
        final_answer.append(active_rules[best_idx].conclusion)
        decision_path.append([active_rules[best_idx]])
      elif tie_breaker_strategy == TiebreakerStrategy.FIRST_HIT_RULE:
        first_hit_idx = int(np.argmax(row_mask))
        final_answer.append(Y_hat[i, first_hit_idx])
        decision_path.append([self.rules[first_hit_idx]])
      else:
        raise ValueError(f"Tie breaker strategy {tie_breaker_strategy} is not supported.")
    return np.array(final_answer), decision_path
  
  
  def predict_numpy_rules(self, 
                          X: np.ndarray, 
                          tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE,
                          return_decision_path: bool = False) -> Any:
    """Generates predictions based on the complete feature numpy array.

    :param X: Complete feature array to be evaluated.
    :type X: np.ndarray
    :param tie_breaker_strategy: Strategy to break ties between rules, defaults to TiebreakerStrategy.FIRST_HIT_RULE
    :type tie_breaker_strategy: TiebreakerStrategy, optional
    :param return_decision_path: Boolean value to return the decision path lead to decision, defaults to False
    :type return_decision_path: bool, optional
    :return: Set of prediction one per row in the feature matrix X.
    :rtype: Any
    """
    if len(self.rules) == 0:
      default_prediction = np.full(X.shape[0], self.defaultRule(), dtype=object)
      default_path = [["default_rule"] for _ in range(X.shape[0])]
      if return_decision_path:
        return default_prediction, default_path
      return default_prediction
    partial_answer = []
    activation_mask = []
    for rule in self.rules:
      pred, mask = rule.predict(X, return_mask=True)
      partial_answer.append(pred)
      activation_mask.append(mask)
    Y_hat = np.array(partial_answer, dtype=object).T
    activation_mask = np.array(activation_mask, dtype=bool).T
    final_decision, decision_path = self.answer_preprocessor(
      Y_hat,
      activation_mask=activation_mask,
      tie_breaker_strategy=tie_breaker_strategy
    )
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
        raise ValueError(f"No elements selected, indexes={col_index}, data={data_row}")
      if res:
        #print(f"answer: {res}")
        ans.append(res)
        active_rules.append(idx_rule)
        # check one condition
        if tie_breaker_strategy == TiebreakerStrategy.FIRST_HIT_RULE:
          return ans, active_rules
    if tie_breaker_strategy == TiebreakerStrategy.MINORITE_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      min_class = classes[np.argmin(counts)]
      return min_class, active_rules
    elif tie_breaker_strategy == TiebreakerStrategy.HIGH_COVERAGE and len(ans)>0:
      max_coverage = -2
      best_idx = -1
      for idx, rule_idx in enumerate(active_rules):
        active_rule = self.rules[rule_idx]
        if active_rule.coverage is not None:
          if active_rule.coverage > max_coverage:
            max_coverage = active_rule.coverage
            best_idx = idx
      if best_idx > -1:
        return ans[best_idx], [active_rules[best_idx ]]
      else:
        return [], []
    elif tie_breaker_strategy == TiebreakerStrategy.MAJORITY_CLASS and len(ans)>0:
      classes, counts = np.unique(ans, return_counts=True)
      max_class = classes[np.argmax(counts)]
      return max_class, active_rules
    elif tie_breaker_strategy == TiebreakerStrategy.HIGH_PERFORMANCE and len(ans)>0:
      max_performance = -2
      best_idx = -1
      for idx, rule_idx in enumerate(active_rules):
        active_rule = self.rules[rule_idx]
        if active_rule.proba is not None:
          if active_rule.proba > max_performance:
            max_performance = active_rule.proba
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
      raise ValueError(f"Input cannot have rank over 2, current rank shape: {shape}")
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
             X: np.ndarray, 
             y_true: np.ndarray, 
             evaluation_method: Dict[str, Callable] = None, 
             mode: Mode = Mode.CLASSIFICATION) -> Dict[str, float]:
    """Evaluates the rule set given a numpy array. 

    :param X: Complete feature array. 
    :type X: np.ndarray
    :param y_true: Ground truth values. 
    :type y_true: np.ndarray
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
        raise ValueError(f"Mode {mode} not supported")
    y_pred = self.predict_numpy_rules(X)
    for key in evaluation_method.keys():
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
      
  def load(self, filename: str) -> "RuleSet":
    """Load a rule set from a file. 

    :param filename: Relative or absolute file path to the binary file should end with ".pkl" extension.
    :type filename: str
    """
    with open(filename, mode='rb') as fp:
      loaded_ruleset = dill.load(fp)
    if not isinstance(loaded_ruleset, RuleSet):
      raise TypeError(f"Loaded object is not a RuleSet: {type(loaded_ruleset)}")
    self.rules = loaded_ruleset.rules
    self.tie_breaker_strategy = loaded_ruleset.tie_breaker_strategy
    self.majority_class = loaded_ruleset.majority_class
    self.print_stats = loaded_ruleset.print_stats
    return self
