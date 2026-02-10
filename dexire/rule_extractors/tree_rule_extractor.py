import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.utils.validation import  check_is_fitted

from ..core.dexire_abstract import Mode, AbstractRuleExtractor, AbstractRuleSet
from ..core.expression import Expr
from ..core.rule import Rule
from ..core.rule_set import RuleSet
from ..core.clause import ConjunctiveClause, DisjunctiveClause

TREE_UNDEFINED = -2

class TreeRuleExtractor(AbstractRuleExtractor):
  """Extract rules based on a decision tree.

  :param AbstractRuleExtractor: Abstract class to rule extraction.
  :type AbstractRuleExtractor: AbstractRuleExtractor
  """
  def __init__(self,
               max_depth: int = 10,
               mode: Mode = Mode.CLASSIFICATION,
               criterion: str = 'gini',
               class_names: List[str] = None,
               min_samples_split: float = 0.1) -> None:
    """Constructor for TreeRuleExtractor.

    :param max_depth: Maximum depth for the decision tree, defaults to 10
    :type max_depth: int, optional
    :param mode: Parameter to choose if it is classification or regression, defaults to Mode.CLASSIFICATION
    :type mode: Mode, optional
    :param criterion: Criterion to split the tree, defaults to 'gini'
    :type criterion: str, optional
    :param class_names: List of class names, defaults to None
    :type class_names: List[str], optional
    :param min_samples_split: Min percentage of samples to split the tree, defaults to 0.1
    :type min_samples_split: float, optional
    :raises Exception: Not implemented mode if it is not Mode.CLASSIFICATION or Mode.REGRESSION.
    """
    self.mode = mode
    self.model = None
    self.max_depth = max_depth
    self.criterion = criterion
    self.class_names = class_names
    self.majority_class = None
    if self.mode == Mode.CLASSIFICATION:
      self.model = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                          min_samples_split=min_samples_split)
    elif self.mode == Mode.REGRESSION:
      self.model = DecisionTreeRegressor(max_depth=self.max_depth, criterion=self.criterion)
    else:
      raise NotImplementedError(f"Mode {self.mode} not implemented")

  def _resolve_class_conclusion(self, class_index: int) -> Any:
    """Resolve the class conclusion label from a class index.

    The decision tree stores class counts according to ``model.classes_`` order.
    Relying only on ``class_names`` positional indexing may invert labels when the
    fitted class order differs from the provided name order.
    """
    if self.model is None:
      raise ValueError("The model has not been defined! model: None")
    if not hasattr(self.model, "classes_"):
      raise ValueError("The model has not been fitted yet and has no classes_.")
    model_classes = self.model.classes_
    predicted_label = model_classes[class_index]
    # If class_names are provided and classes are numeric indexes, map by index.
    # Otherwise, keep the fitted class label to avoid semantic inversion.
    if self.class_names is not None:
      if isinstance(predicted_label, (int, np.integer)):
        if 0 <= int(predicted_label) < len(self.class_names):
          return self.class_names[int(predicted_label)]
      if predicted_label in self.class_names:
        return predicted_label
    return predicted_label

  def get_rules(self, feature_names: List[str]) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Get the rules from the tree model.
    
    :param feature_names: List of feature names.
    :type feature_names: List[str]
    :raises Exception: The model has not been defined! model: None
    :return: extracted rule set.
    :rtype: Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]
    """
    if self.model is not None:
      # Check if the model is fitted
      check_is_fitted(self.model)
      tree_ = self.model.tree_
    else:
      raise ValueError("The model has not been defined! model: None")
    # feature naming
    feature_name = [
        feature_names[i] if i != TREE_UNDEFINED else None
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
      if tree_.feature[node] != TREE_UNDEFINED:
        name = feature_name[node]
        feature_index = tree_.feature[node]
        threshold = tree_.threshold[node]
        p1, p2 = list(path), list(path)
        p1 += [Expr(feature_index, np.round(threshold, 3), '<=', name)]
        # p1 += [f"({name} <= {np.round(threshold, 3)})"]
        recurse(tree_.children_left[node], p1, paths)
        p2 += [Expr(feature_index, np.round(threshold, 3), '>', name)]
        # p2 += [f"({name} > {np.round(threshold, 3)})"]
        recurse(tree_.children_right[node], p2, paths)
      else:
        path += [(tree_.value[node], tree_.n_node_samples[node])]
        paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    # empty rule set
    rs = RuleSet()
    for path in paths:
      # print(f"path: {path[:-1]}")
      # print(f"path: {path}")
      # all rules in a path are join by a conjuntion
      rule_premise = ConjunctiveClause(path[:-1])
      # there is not class names for example in regression
      if self.mode == Mode.REGRESSION:
        # Regression mode there is not classes 
        conclusion = str(np.round(path[-1][0][0][0],3))
        proba = None
      elif self.mode == Mode.CLASSIFICATION:
        classes = path[-1][0][0]
        l = int(np.argmax(classes))
        conclusion = self._resolve_class_conclusion(class_index=l)
        # calculate accuracy probability and coverage of the rule
        proba = np.round(100.0*classes[l]/np.sum(classes),2)
      else: 
        raise NotImplementedError(f"Mode {self.mode} not implemented")
      coverage = path[-1][1]
      # create the rule
      rule = Rule(premise=rule_premise,
                  conclusion=conclusion,
                  proba=proba,
                  coverage=coverage)
      # add the rule to the rule set
      rs.add_rules([rule])
    return rs

  def get_model(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor, None]:
    """Returns the decision tree classifier or regressor model employed to extract the rules.

    :return: The tree model employed to extract the rules. 
    :rtype: Union[DecisionTreeClassifier, DecisionTreeRegressor, None]
    """
    return self.model

  def extract_rules(self, X: Any, y: Any, feature_names: str = None) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    """Train the tree model and extract rules from the dataset (X, y).

    :param X: Input features dataset.
    :type X: Any
    :param y: Labels for dataset X.
    :type y: Any
    :param feature_names: List of feature names, defaults to None.
    :raises Exception: No model. If the tree model has not been defined.
    :raises Exception: The feature list size is different to the number of columns.
    :return: Extracted rule set.
    :rtype: Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]
    """
    if self.model is not None:
      # train the model
      self.model.fit(X, y)
      # extract rules
      if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
      else:
        # Check features size
        if len(feature_names)!= X.shape[1]:
          raise ValueError(f"feature_names size {len(feature_names)}!= X.shape[1] {X.shape[1]}")
      rules = self.get_rules(feature_names=feature_names)
      return rules
    else:
      raise ValueError("No model")