import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set

from ..core.dexire_abstract import Mode, AbstractRuleExtractor, AbstractRuleSet
from ..core.expression import Expr
from ..core.rule import Rule
from ..core.rule_set import RuleSet
from ..core.clause import ConjunctiveClause, DisjunctiveClause


class OneRuleExtractor(AbstractRuleExtractor):
  """Extract rules using OneRule extractor algorithm and sequential coverage. 

  :param AbstractRuleExtractor: Abstract class for rule extraction.
  :type AbstractRuleExtractor: AbstractRuleExtractor
  """

  def __init__(self, 
               majority_class:Any=None,
               discretize:bool=False,
               columns_to_discretize:List[int]=None,
               mode=Mode.CLASSIFICATION,
               precision_decimal:int=4,
               minimum_coverage:Union[int, float]=0.1,
               minimum_accuracy: float = 0.51, 
               feature_quantization_resolution: int = 5,
               regression_resolution_quantization: int = 255,
               multi_class_acc_threshold: float = 0.85,
               max_iteration: int = 10000) -> None:
    """Constructor for OneRuleExtractor.

    :param features_names:List of feature names to include in the rules, defaults to None
    :type features_names: List[str], optional
    :param majority_class: the class with more samples in the data, defaults to None
    :type majority_class: Any, optional
    :param discretize: True for discretize the input, defaults to False
    :type discretize: bool, optional
    :param columns_to_discretize: List of columns to be discretized, defaults to None
    :type columns_to_discretize: List[int], optional
    :param mode: If the rule extraction is classification or regression, defaults to Mode.CLASSIFICATION
    :type mode: _type_, optional
    :param precision_decimal: Number of significant figures to be included in the rule, defaults to 4
    :type precision_decimal: int, optional
    :param minimum_coverage: Minimum percentage of sample that a rule should cover, defaults to 0.1
    :type minimum_coverage: Union[int, float], optional
    :param minimum_accuracy: Minimum accuracy that a rule should reach, defaults to 0.51
    :type minimum_accuracy: float, optional
    """
    self.rules = []
    self.majority_class = majority_class
    self.mode = mode
    self.multi_class_acc_threshold = multi_class_acc_threshold
    self.regression_resolution_quantization = regression_resolution_quantization
    self.regression_bins = None
    self.X = None
    self.y = None
    self.bins_dict = {}
    self.discretize = discretize
    self.columns_to_discretize = columns_to_discretize
    self.feature_quantization_resolution = feature_quantization_resolution
    self.precision_decimal = precision_decimal
    self.minimum_coverage = minimum_coverage
    self.minimum_accuracy = minimum_accuracy
    self.max_iterations = max_iteration

  def preprocessing_data(self, X, y) -> None:
    """Preprocess input features with discretize if discretize is True or a list of columns to digitalize is provided.
    """
    x = X.copy()
    y_final = y.copy()
    if self.discretize:
      if self.columns_to_discretize is not None:
        for col_idx in self.columns_to_discretize:
          digitalized_column, bins = self.digitalize_column(x, 
                                                            col_idx,
                                                            n_bins=self.feature_quantization_resolution)
          x[:, col_idx] = digitalized_column
          self.bins_dict[col_idx] = bins
      else:
        for col_idx in range(self.X.shape[1]):
          digitalized_column, bins = self.digitalize_column(x, 
                                                            col_idx, 
                                                            n_bins=self.feature_quantization_resolution)
          x[:, col_idx] = digitalized_column
          self.bins_dict[col_idx] = bins
    # in case of regression
    if self.mode == Mode.REGRESSION:
      y_final = y.reshape(-1, 1)
      if self.regression_resolution_quantization <= 2:
        raise Exception("The regression_resolution_quantization must be greater than 2.")
      # transform the regression input 
      y_final, self.regression_bins = self.digitalize_column(y_final, 
                                                            col_idx=0, 
                                                            bins=self.regression_bins,
                                                            n_bins=self.regression_resolution_quantization) 
    return x, y_final
      

  def post_processing_rules(self) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Transform discretized rules to original values. 

    :return: rules defined in original feature values. 
    :rtype: Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]
    """
    transformed_rules = []
    for rule in self.rules:
      if rule.premise.feature_idx in self.bins_dict.keys():
        transformed_rule = self.transform_rule(rule)
      else:
        transformed_rule = rule
      transformed_rules.append(transformed_rule)
    return transformed_rules
  
  def process_complex_premises(self, premise: Expr) -> Expr:
    """Transform complex premises in to clauses.

    :param premise: premise to be transformed.
    :type premise: Expr
    :return: Processed premise.
    :rtype: Expr
    """
    if isinstance(premise, Expr):
        return self.inverse_digitalize_exp(premise,
                                          self.bins_dict[premise.feature_idx])
    else:
      transformed_expressions = []
      expressions = premise.clauses
      for ex in expressions:
        transformed_expressions.append(self.process_complex_premises(ex))
        if isinstance(premise, ConjunctiveClause):
          transformed_premise = ConjunctiveClause(transformed_expressions)
        elif isinstance(premise, DisjunctiveClause):
          transformed_premise = DisjunctiveClause(transformed_expressions)
      return transformed_premise

  def transform_rule(self, rule: Rule) -> Rule:
    """Transform one rule to original input values.

    :param rule: Rule to be transformed.
    :type rule: Rule
    :return: Transformed rule. 
    :rtype: Rule
    """
    if self.discretize:
      transformed_premise = self.process_complex_premises(rule.premise)
    else:
      transformed_premise = rule.premise
    if self.mode == Mode.REGRESSION and self.regression_bins is not None:
      transformed_conclusion = self.inverse_transform_target(rule.conclusion,
                                                          self.regression_bins)
    else:
      transformed_conclusion = rule.conclusion
    transformed_rule = Rule(premise=transformed_premise,
                            conclusion=transformed_conclusion,
                            proba=rule.proba,
                            accuracy=rule.accuracy,
                            coverage=rule.coverage)
    return transformed_rule

  def get_rules(self) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Return the list pf extracted rules.

    :return: List of extracted rules. 
    :rtype: Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]
    """
    return self.rules

  def digitalize_column(self, 
                        X: np.array, 
                        col_idx: int, 
                        bins:List[Any]=None, 
                        n_bins:int=10) -> Tuple[np.ndarray, List[Any]]:
    """Digitalize one column of the input array.

    :param X: Numpy array with the input features.
    :type X: np.array
    :param col_idx: Column to be discretized in X. 
    :type col_idx: int
    :param bins: List of bins to discretized column identified by col_idx, defaults to None
    :type bins: List[Any], optional
    :param n_bins: Bins number, defaults to 10
    :type n_bins: int, optional
    :return: discretized column and bins. 
    :rtype: Tuple[np.ndarray, List[Any]]
    """
    temp_x = X[:, col_idx]
    if bins is None:
      max_val = np.max(temp_x)
      min_val = np.min(temp_x)
      bins = np.linspace(min_val, max_val, n_bins)
    digitalized_column = np.digitize(temp_x, bins)
    return digitalized_column, bins

  def inverse_transform_target(self,
                               digitalized_value: float,
                               bins:List[Any]) -> float:
    """Transform a digitalized value to original value.

    :param digitalized_value: Digitalized value.
    :type digitalized_value: float
    :param bins: Bin list used to generated the digitalized expression.
    """
    int_threshold = int(digitalized_value)-1
    lower = bins[int_threshold-1]
    higher = bins[int_threshold]
    return np.round(np.mean([lower,higher]), self.precision_decimal)
    

  def inverse_digitalize_exp(self, 
                             digitalize_expr:Expr, 
                             bins:List[Any],) -> ConjunctiveClause:
    """Transform a digitalized expression to original input values.

    :param digitalize_expr: Digitalized expression.
    :type digitalize_expr: Expr
    :param bins: Bin list used to generated the digitalized expression.
    :type bins: List[Any]
    :return: Expression with original input values. 
    :rtype: ConjunctiveClause
    """
    int_threshold = int(digitalize_expr.threshold)-1
    print(f"int t: {int_threshold}")
    print(f"bins: {bins}")
    lower = bins[int_threshold-1]
    higher = bins[int_threshold]
    expr1 = Expr(digitalize_expr.feature_idx,
                np.round(lower, self.precision_decimal),
                '>=',
                 digitalize_expr.feature_name)
    expr2 = Expr(digitalize_expr.feature_idx,
                np.round(higher, self.precision_decimal),
                '<',
                 digitalize_expr.feature_name)
    return ConjunctiveClause([expr1, expr2])

  def remove_covered_examples(self, 
                              X: np.array, 
                              y: np.array, 
                              covered_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Removes the covered examples from the dataset.

    :param X: Input features dataset.
    :type X: np.array
    :param y: labels for dataset X.
    :type y: np.array
    :param covered_indices: List of covered indices for the current rule.
    :type covered_indices: List[int]
    :return: Dataset without covered examples. 
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    mask = np.ones(len(X), dtype=bool)
    mask[covered_indices] = False
    return X[mask], y[mask]

  def oneR(self, 
           X: np.array, 
           y: np.array, 
           col_dim:int =1,
           feature_names: List[str]=None
           ) -> Tuple[Rule, np.ndarray]:
    """Extract one rule from the dataset (X, y).

    :param X: Input feature dataset.
    :type X: np.array
    :param y: Labels for dataset X.
    :type y: np.array
    :param col_dim: Dimension to get column index , defaults to 1
    :type col_dim: int, optional
    :return: The learned rule and indices of the covered examples.
    :rtype: Tuple[Rule, np.ndarray]
    """
    # get unique values
    best_rule = None
    best_covered_indices = None
    rule_error = np.inf
    # iterate over unique values of the column
    print(f"unique: {len(np.unique(X[:, col_dim]))}")
    for i in range(X.shape[col_dim]):
      temp_x = X[:, i]
      unique_values = np.unique(temp_x)
      print(f"unique: {len(unique_values)}")
      for val in unique_values:
        condition_idx = np.where(temp_x == val)[0]
        labels, counts = np.unique(y[condition_idx], return_counts=True)
        # get conclusion
        conclusion = labels[np.argmax(counts)]
        # calculate coverage
        coverage = np.sum(temp_x == val)/X.shape[0]
        # get accuracy
        accuracy = np.round(100.0*np.max(counts)/np.sum(counts), 2)
        error=100.0-accuracy
        # check if the rule is better
        if error <= rule_error:
          # create rule
          if feature_names is not None:
            predicate = Expr(feature_idx=i,
                             threshold=val, 
                             operator='==',
                            feature_name=feature_names[i])
          else:
            predicate = Expr(i, val, '==')
          best_rule = Rule(predicate, conclusion, accuracy, coverage)
          best_covered_indices = condition_idx
          rule_error = error
    # check if the rule is better
    return best_rule, best_covered_indices
      

  # remove covered examples
  def sequential_covering_oneR(self, X: np.array, y: np.array, feature_names: List[str] = None) -> None:
    """Iterates over the dataset and extracts rules.

    :param X: Input features dataset.
    :type X: np.array
    :param y: Labels for dataset X.
    :type y: np.array
    """
    accuracy = np.inf
    coverage = np.inf
    iterations = 0
    # check_multi-class case 
    multi_class_flag = False
    if self.mode == Mode.CLASSIFICATION:
      multi_class_flag = len(np.unique(y)) > 2
    while len(X) > 0 and\
      accuracy >= self.minimum_accuracy and\
      coverage >= self.minimum_coverage and\
      iterations < self.max_iterations:
      print(f"--------- Iter = {iterations}-----------")
      rule, covered_indices = self.oneR(X, y, feature_names=feature_names)
      self.rules.append(rule)
      accuracy = rule.accuracy
      coverage = rule.coverage
      X, y = self.remove_covered_examples(X, y, covered_indices)
      iterations += 1

  def extract_rules(self, X: Any, y: Any, feature_names:List[str]=None) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Extract rules from the dataset (X, y).

    :param X: Input features dataset.
    :type X: np.array
    :param y: Labels for dataset X.
    :type y: np.array
    :return: Learned rule set. 
    :rtype: Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]
    """
    self.X = X
    self.y = y
    # check feature names 
    if feature_names is not None:
      if len(feature_names) != X.shape[1]:
        raise ValueError("The number of feature names must be equal\
          to the number of feature columns")
    else:
      feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    rs = RuleSet()
    X_new, y_new = self.preprocessing_data(X, y)
    print(f"X: {X_new}")
    self.sequential_covering_oneR(X_new, y_new, feature_names=feature_names)
    self.rules = self.post_processing_rules()
    rs.add_rules(self.rules)
    return rs