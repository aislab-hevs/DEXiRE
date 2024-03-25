import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import _tree

from ..core.dexire_abstract import Mode, AbstractRuleExtractor, AbstractRuleSet
from ..core.expression import Expr
from ..core.rule import Rule
from ..core.rule_set import RuleSet
from ..core.clause import ConjuntiveClause, DisjuntiveClause

class TreeRuleExtractor(AbstractRuleExtractor):
  def __init__(self,
               max_depth: int = 10,
               mode: Mode = Mode.CLASSIFICATION,
               criterion: str = 'gini',
               features_names: List[str] = None,
               class_names: List[str] = None,
               min_samples_split: float = 0.1) -> None:
    self.mode = mode
    self.model = None
    self.max_depth = max_depth
    self.criterion = criterion
    self.feature_names = features_names
    self.class_names = class_names
    self.majority_class = None
    if self.mode == Mode.CLASSIFICATION:
      self.model = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                          min_samples_split=min_samples_split)
    elif self.mode == Mode.REGRESSION:
      self.model = DecisionTreeRegressor(max_depth=self.max_depth, criterion=self.criterion)
    else:
      raise Exception("Mode not implemented")

  def get_rules(self):
    if self.model is not None:
      tree_ = self.model.tree_
    else:
      raise Exception("The model has not been defined! model: None")
    # feature naming
    feature_name = [
        self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
      if tree_.feature[node] != _tree.TREE_UNDEFINED:
        print(tree_.feature[node])
        name = feature_name[node]
        print(name)
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
      rule_premise = ConjuntiveClause(path[:-1])
      # there is not class names for example in regression
      if self.class_names is None:
        conclusion = "response: "+str(np.round(path[-1][0][0][0],3))
      else:
        # there is class names
        classes = path[-1][0][0]
        l = np.argmax(classes)
        conclusion = f"class: {class_names[l]}"
      # calculate accuracy probability and coverage of the rule
      proba = np.round(100.0*classes[l]/np.sum(classes),2)
      coverage = path[-1][1]
      # create the rule
      rule = Rule(premise=rule_premise,
                  conclusion=conclusion,
                  proba=proba,
                  coverage=coverage)
      # add the rule to the rule set
      rs.add_rules([rule])

    return rs

  def get_model(self):
    return self.model

  def extract_rules(self, X: Any, y: Any) -> Union[AbstractRuleSet, Set[AbstractRuleSet], List[AbstractRuleSet], None]:
    if self.model is not None:
      # train the model
      self.model.fit(X, y)
      # extract rules
      if self.feature_names is None:
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
      rules = self.get_rules()
      return rules
    else:
      raise Exception("No model")