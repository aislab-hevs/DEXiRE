import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .core.dexire_abstract import AbstractRuleExtractor, AbstractRuleSet, Mode, RuleExtractorEnum
from .rule_extractors.tree_rule_extractor import TreeRuleExtractor
from .rule_extractors.one_rule_extractor import OneRuleExtractor
from .core.rule_set import RuleSet
from .core.dexire_abstract import AbstractRuleExtractor, AbstractRuleSet


class DEXiRE:
  """Deep Explanations and Rule Extraction pipeline to extract rules from a deep neural network.
  """
  def __init__(self, 
               model: tf.keras.Model, 
               feature_names: List[str]=None, 
               class_names: List[str]=None,
               rule_extractor: Union[str, AbstractRuleExtractor]=RuleExtractorEnum.TREERULE,
               mode: Mode = Mode.CLASSIFICATION) -> None:
    """Constructor method to set up the DEXiRE pipeline.

    :param model: Trained deep learning model to explain it (to extract rules from).
    :type model: tf.keras.Model
    :param feature_names: List of feature names, defaults to None
    :type feature_names: List[str], optional
    :param class_names: Target class names, defaults to None
    :type class_names: List[str], optional
    """
    self.model = model
    self.mode = mode
    self.rule_extractor = rule_extractor
    self.features_names = feature_names
    self.class_names = class_names
    self.intermediate_model = None
    self.data_raw = {}
    self.data_transformed = {}
    if not issubclass(self.rule_extractor, AbstractRuleExtractor) \
      and issubclass(self.rule_extractor, str):
      # Check modes 
      if self.mode!= Mode.CLASSIFICATION and self.mode!= Mode.REGRESSION:
        raise Exception(f"Not implemented mode: {self.mode} if it is not Mode.CLASSIFICATION or Mode.REGRESSION.")
      # Check if the name of rule extractor is registered 
      if self.rule_extractor not in RuleExtractorEnum:
        raise Exception("Rule extractor not implemented")
      elif self.rule_extractor == RuleExtractorEnum.ONERULE:
        self.rule_extractor = OneRuleExtractor(
          features_names=self.features_names,
          mode=self.mode
        )
      elif self.rule_extractor == RuleExtractorEnum.TREERULE:
        self.rule_extractor = TreeRuleExtractor(max_depth=200, 
                                                mode=self.mode,
                                                features_names=self.features_names,
                                                class_names = self.class_names)
      elif self.rule_extractor == RuleExtractorEnum.MIXED:
        self.rule_extractor = {
          "oneR": OneRuleExtractor(
            features_names=self.features_names,
            mode=self.mode
          ),
          "treeR": TreeRuleExtractor(max_depth=200, 
                                    mode=self.mode,
                                    features_names=self.features_names,
                                    class_names = self.class_names)
        }

  def get_intermediate_model(self, layer_idx: int) -> tf.keras.Model:
    """Get intermediate model from the deep learning model.

    :param layer_idx: layer index to get the intermediate model.
    :type layer_idx: int
    :return: Intermediate model at a given layer index.
    :rtype: tf.keras.Model
    """
    intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                 outputs=self.model.layers[layer_idx].output)
    return intermediate_layer_model

  def get_raw_data(self) -> Dict[str, Any]:
    """Get original input data.

    :return: Get original data used to train the model.
    :rtype: Dict[str, Any]
    """
    return self.data_row

  def get_data_transformed(self) -> Dict[str, np.array]:
    """Get transformed input data at each layer of the model.

    :return: Transformed input data at each layer of the model.
    :rtype: Dict[str, np.array]
    """
    return self.data_transformed

  def extract_rules(self, 
                    X:np.array =None, 
                    y:np.array =None, 
                    layer_idx: int = -2, 
                    sample=None) -> List[AbstractRuleSet]:
    """Extract rules from a deep neural network.

    :param X: Input features dataset, defaults to None
    :type X: np.array, optional
    :param y: Labels for dataset X, defaults to None
    :type y: np.array, optional
    :param layer_idx: Index of first hidden layer, defaults to -2
    :type layer_idx: int, optional
    :param sample: sample percentage to extract rules from if None all examples will be used, defaults to None
    :type sample: _type_, optional
    :return: Rule set extracted from the deep neural network.
    :rtype: List[AbstractRuleSet]
    """
    self.data_raw['inputs'] = X
    self.data_raw['output'] = y

    y_pred_raw = self.model.predict(X)
    y_pred = np.rint(y_pred_raw)
    classes, counts = np.unique(y_pred, return_counts=True)
    self.majority_class = classes[np.argmax(counts)]
    print(f"Number of classes: {classes}")
    if self.intermediate_model is None:
      self.intermediate_model = self.get_intermediate_model(layer_idx=layer_idx)
    intermediate_output = self.intermediate_model.predict(X)
    x = intermediate_output
    y = y_pred
    self.data_transformed['inputs'] = x
    self.data_transformed['output'] = y
    if sample is not None:
      _, x, _,y = train_test_split(intermediate_output, y_pred, test_size=sample, stratify=y_pred)
      print(f"sample shape: {x.shape} label sample: {y.shape}")
    rules = []
    rules = self.rule_extractor.extract_rules(x, y)
    return rules
