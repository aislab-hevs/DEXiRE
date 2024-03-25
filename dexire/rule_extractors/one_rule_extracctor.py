import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set

from ..core.dexire_abstract import Mode, AbstractRuleExtractor, AbstractRuleSet
from ..core.expression import Expr
from ..core.rule import Rule
from ..core.rule_set import RuleSet
from ..core.clause import ConjuntiveClause, DisjuntiveClause


class OneRuleExtractor(AbstractRuleExtractor):

  def __init__(self, features_names:List[str]=None,
               majority_class:Any=None,
               discretize:bool=False,
               columns_to_discretize:List[int]=None,
               mode=Mode.CLASSIFICATION,
               precision_decimal:int=4,
               minimum_coverage:Union[int, float]=0.1,
               minimum_accuracy: float = 0.51):
    self.rules = []
    self.features_names = features_names
    self.majority_class = majority_class
    self.mode = mode
    self.X = None
    self.y = None
    self.bins_dict = {}
    self.discretize = discretize
    self.columns_to_discretize = columns_to_discretize
    self.precision_decimal = precision_decimal
    self.minimum_coverage = minimum_coverage
    self.minimum_accuracy = minimum_accuracy

  def preprocessing_data(self):
    if self.columns_to_discretize is not None:
      for col_idx in self.columns_to_discretize:
        digitalized_column, bins = self.digitalize_column(self.X, col_idx)
        self.X[:, col_idx] = digitalized_column
        self.bins_dict[col_idx] = bins
    else:
      for col_idx in range(self.X.shape[1]):
        digitalized_column, bins = self.digitalize_column(self.X, col_idx)
        self.X[:, col_idx] = digitalized_column
        self.bins_dict[col_idx] = bins

  def post_processing_rules(self):
    transformed_rules = []
    for rule in self.rules:
      if rule.premise.feature_idx in self.bins_dict.keys():
        transformed_rule = self.transform_rule(rule)
      else:
        transformed_rule = rule
      transformed_rules.append(transformed_rule)
    return transformed_rules

  def transform_rule(self, rule):
    transformed_premise = self.inverse_digitalize_exp(rule.premise,
                                                      self.bins_dict[rule.premise.feature_idx])
    transformed_rule = Rule(premise=transformed_premise,
                            conclusion=rule.conclusion,
                            proba=rule.proba,
                            accuracy=rule.accuracy,
                            coverage=rule.coverage)
    return transformed_rule

  def get_rules(self):
    return self.rules

  def digitalize_column(self, X, col_idx, bins:List[Any]=None, n_bins:int=10):
    temp_x = X[:, col_idx]
    if bins is None:
      max_val = np.max(temp_x)
      min_val = np.min(temp_x)
      bins = np.linspace(min_val, max_val, n_bins)
    digitalized_column = np.digitize(temp_x, bins)
    return digitalized_column, bins

  def inverse_digitalize_exp(self, digitalize_expr:Expr, bins:List[Any]):
    int_threshold = int(digitalize_expr.threshold)
    lower = bins[int_threshold-1]
    higher = bins[int_threshold]
    expr1 = Expr(digitalize_expr.feature_idx,
                np.round(lower, self.precision_decimal),
                '<=',
                 digitalize_expr.feature_name)
    expr2 = Expr(digitalize_expr.feature_idx,
                np.round(higher, self.precision_decimal),
                '>',
                 digitalize_expr.feature_name)
    return ConjuntiveClause([expr1, expr2])

  def remove_covered_examples(self, X, y, covered_indices):
    mask = np.ones(len(X), dtype=bool)
    mask[covered_indices] = False
    return X[mask], y[mask]

  def oneR(self, X, y, col_idx=1):
    # get unique values
    best_rule = None
    best_coverage = -1
    best_covered_indices = None
    rule_error = np.inf
    for i in range(X.shape[col_idx]):
      temp_x = X[:, i]
      unique_values = np.unique(temp_x)
      for val in unique_values:
        condition_idx = np.where(temp_x == val)[0]
        labels, counts = np.unique(y[condition_idx], return_counts=True)
        # create rule
        if self.features_names is not None:
          predicate = Expr(self.features_names[i],
                           val, '==',
                           self.features_names[i])
        else:
          predicate = Expr(i, val, '==')
        # get conclusion
        conclusion = labels[np.argmax(counts)]
        # calculate coverage
        coverage = np.sum(y[condition_idx] == conclusion)
        # get accuracy
        accuracy = np.round(100.0*np.max(counts)/np.sum(counts), 2)
        error=100.0-accuracy
        # check if the rule is better
        if coverage > best_coverage and error <= rule_error:
          best_rule = Rule(predicate, conclusion, accuracy, coverage)
          best_covered_indices = condition_idx
          best_coverage = coverage
          rule_error = error
    return best_rule, best_covered_indices

  # remove covered examples
  def sequential_covering_oneR(self, X, y):
    accuracy = np.inf
    coverage = np.inf
    while len(X) > 0 and\
     accuracy >= self.minimum_accuracy and\
      coverage >= self.minimum_coverage:
      rule, covered_indices = self.oneR(X, y)
      self.rules.append(rule)
      accuracy = rule.accuracy
      coverage = rule.coverage
      X, y = self.remove_covered_examples(X, y, covered_indices)

  def extract_rules(self, X: Any, y: Any) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    self.X = X
    self.y = y
    rs = RuleSet()
    if self.discretize:
      self.preprocessing_data()
      self.sequential_covering_oneR(self.X, self.y)
      self.rules = self.post_processing_rules()
    else:
      self.sequential_covering_oneR(self.X, self.y)
    rs.add_rules(self.rules)
    return rs