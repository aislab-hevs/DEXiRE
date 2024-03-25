import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model

from .rule_extractors.tree_rule_extractor import TreeRuleExtractor


class DEXiRE:
  def __init__(self, model, feature_names: List[str]=None, class_names: List[str]=None):
    self.model = model
    self.features_names = feature_names
    self.class_names = class_names
    self.rule_extractor = TreeRuleExtractor(max_depth=200, features_names=self.features_names,
                                            class_names = self.class_names)
    self.intermediate_model = None
    self.data_raw = {}
    self.data_transformed = {}

  def get_intermediate_model(self, layer_idx: int):
    intermediate_layer_model = Model(inputs=self.model.input,
                                 outputs=self.model.layers[layer_idx].output)
    return intermediate_layer_model

  def get_raw_data(self):
    return self.data_row

  def get_data_transformed(self):
    return self.data_transformed

  def extract_rules(self, X, y:np.array =None, layer_idx: int = -2, sample=None):
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
