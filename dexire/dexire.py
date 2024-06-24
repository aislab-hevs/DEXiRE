import numpy as np
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .core.dexire_abstract import (AbstractRuleExtractor, 
                                   AbstractRuleSet, 
                                   Mode, 
                                   RuleExtractorEnum,
                                   TiebreakerStrategy)
from .rule_extractors.tree_rule_extractor import TreeRuleExtractor
from .rule_extractors.one_rule_extractor import OneRuleExtractor
from .core.rule_set import RuleSet
from .core.dexire_abstract import AbstractRuleExtractor, AbstractRuleSet
from .utils.activation_discretizer import discretize_activation_layer


class DEXiRE:
  """Deep Explanations and Rule Extraction pipeline to extract rules from a deep neural network.
  """
  def __init__(self, 
               model: tf.keras.Model, 
               feature_names: List[str]=None, 
               class_names: List[str]=None,
               rule_extractor: Union[None, AbstractRuleExtractor] = None,
               mode: Mode = Mode.CLASSIFICATION,
               explain_features: np.array = None,
               rule_extraction_method: RuleExtractorEnum = RuleExtractorEnum.TREERULE,
               tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE ) -> None:
    """Constructor method to set up the DEXiRE pipeline.

    :param model: Trained deep learning model to explain it (to extract rules from).
    :type model: tf.keras.Model
    :param feature_names: List of feature names, defaults to None
    :type feature_names: List[str], optional
    :param class_names: Target class names, defaults to None
    :type class_names: List[str], optional
    """
    self.model = model
    self.explain_features = explain_features
    self.mode = mode
    self.rule_extraction_method = rule_extraction_method
    self.tie_breaker_strategy = tie_breaker_strategy
    self.rule_extractor = rule_extractor
    self.features_names = feature_names
    self.class_names = class_names
    self.intermediate_rules = {}
    self.intermediate_model = {}
    self.data_raw = {}
    self.final_rule_set = None
    self.data_transformed = {}
    if self.rule_extractor is None:
      # Check modes 
      if self.mode!= Mode.CLASSIFICATION and self.mode!= Mode.REGRESSION:
        raise Exception(f"Not implemented mode: {self.mode} if it is not Mode.CLASSIFICATION or Mode.REGRESSION.")
      # Check if the name of rule extractor is registered 
      if self.rule_extraction_method not in RuleExtractorEnum:
        raise Exception(f"Rule extractor: {self.rule_extraction_method} not implemented.")
      elif self.rule_extraction_method == RuleExtractorEnum.ONERULE:
        self.rule_extractor = {RuleExtractorEnum.ONERULE :OneRuleExtractor(
          mode=self.mode
        )}
      elif self.rule_extraction_method == RuleExtractorEnum.TREERULE:
        self.rule_extractor = {RuleExtractorEnum.TREERULE :TreeRuleExtractor(max_depth=200, 
                                                mode=self.mode,
                                                class_names = self.class_names)}
      elif self.rule_extraction_method == RuleExtractorEnum.MIXED:
        self.rule_extractor = {
          RuleExtractorEnum.ONERULE: OneRuleExtractor(
            mode=self.mode
          ),
          RuleExtractorEnum.TREERULE: TreeRuleExtractor(max_depth=200, 
                                    mode=self.mode,
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

  def extract_rules_at_layer(self, 
                    X:np.array =None, 
                    y:np.array =None, 
                    layer_idx: int = -2, 
                    sample: float=None, 
                    quantize: bool = True,
                    n_bins: int = 2,
                    express_as_basic_features: bool = True, 
                    random_state: int = 41) -> List[AbstractRuleSet]:
    """Extract rules from a deep neural network.

    :param X: Input features dataset, defaults to None
    :type X: np.array, optional
    :param y: Labels for dataset X, defaults to None
    :type y: np.array, optional
    :param layer_idx: Index of first hidden layer, defaults to -2
    :type layer_idx: int, optional
    :param sample: sample percentage to extract rules from if None all examples will be used, defaults to None
    :type sample: _type_, optional
    :param quantize: Quantize activations, defaults to True
    :type quantize: bool, optional
    :param n_bins: Number of bins to discretize activations, defaults to 2
    :type n_bins: int, optional
    :param express_as_basic_features: Express the current rule as basic features, defaults to True
    :type express_as_basic_features: bool, optional.
    :return: Rule set extracted from the deep neural network.
    :rtype: List[AbstractRuleSet]
    """
    self.data_raw['inputs'] = X
    self.data_raw['output'] = y
    # retrieve prediction 
    if "raw_prediction" in self.data_raw.keys():
      y_pred_raw = self.data_raw['raw_prediction']
    else: 
      y_pred_raw = self.model.predict(X)
      self.data_raw['raw_prediction'] = y_pred_raw
    # Transform the predictions
    if self.mode == Mode.CLASSIFICATION:
      pred_shape = y_pred_raw.shape[1]
      if pred_shape == 1:
        # binary classification 
        y_pred = np.rint(y_pred_raw)
      elif pred_shape > 1:
        y_pred = np.argmax(y_pred_raw, axis=1)
      else:
        raise Exception(f"The prediction shape cannot be processed expected (batch, n), obtained: {y_pred_raw.shape}")
      print(f"Unique predictions: {np.unique(y_pred)}")
      classes, counts = np.unique(y_pred, return_counts=True)
      self.majority_class = classes[np.argmax(counts)]
    elif self.mode == Mode.REGRESSION:
      y_pred = y_pred_raw
      self.majority_class = np.mean(y_pred)
    else:
      raise Exception(f"Not implemented mode: {self.mode} if it is not Mode.CLASSIFICATION or Mode.REGRESSION.")
    
    if layer_idx not in self.intermediate_model.keys():
      self.intermediate_model[layer_idx] = self.get_intermediate_model(layer_idx=layer_idx)
    intermediate_output = self.intermediate_model[layer_idx].predict(X)
    # quantize activations if it is required 
    if quantize:
      intermediate_output = discretize_activation_layer(intermediate_output, 
                                                        n_bins=n_bins)
    # Intermediate data 
    x = intermediate_output
    y = y_pred
    # sample data if the dataset es big
    if sample is not None and self.mode == Mode.CLASSIFICATION:
      _, x, _,y = train_test_split(x, y, test_size=sample, stratify=y, 
                                   random_state=random_state)
    elif sample is not None and self.mode == Mode.REGRESSION:
      _, x, _,y = train_test_split(x, y, test_size=sample, 
                                   random_state=random_state)
    rules = []
    if self.rule_extraction_method == RuleExtractorEnum.ONERULE:
      rules = self.rule_extractor[RuleExtractorEnum.ONERULE].extract_rules(x, y)
    elif self.rule_extraction_method == RuleExtractorEnum.TREERULE:
      rules = self.rule_extractor[RuleExtractorEnum.TREERULE].extract_rules(x, y)
    elif self.rule_extraction_method == RuleExtractorEnum.MIXED:
      rules = self.rule_extractor[RuleExtractorEnum.ONERULE].extract_rules(x, y)
    # transform intermediate rule set into features expression
    y_rule = rules.predict_numpy_rules(x, 
                                       tie_breaker_strategy=self.tie_breaker_strategy)
    # generate feature based rules 
    if express_as_basic_features:
      if self.explain_features is not None:
        X_xai = self.explain_features
      else:
        X_xai = X
      # check feature name with xai_features 
      if self.features_names is not None:
        if len(self.features_names) != X_xai.shape[1]:
          raise ValueError(f"The feature names list length do not coincide with XAI features columns. \
            Expected values {X_xai.shape[1]}, provided: {len(self.features_names)}.")
      else:
        self.features_names = [f"X_{i}" for i in range(X_xai.shape[1])]
      # use given features to generate rules 
      if self.rule_extraction_method == RuleExtractorEnum.MIXED:
        rules_features = self.rule_extractor[RuleExtractorEnum.TREERULE].extract_rules(X_xai, 
                                                                                       y_rule,
                                                                                       feature_names=self.features_names)
      elif self.rule_extraction_method == RuleExtractorEnum.ONERULE:
        rules_features = self.rule_extractor[RuleExtractorEnum.ONERULE].extract_rules(X_xai, 
                                                                                      y_rule, 
                                                                                      feature_names=self.features_names)
      else:
        rules_features = self.rule_extractor[RuleExtractorEnum.TREERULE].extract_rules(X_xai, 
                                                                                       y_rule,
                                                                                       feature_names=self.features_names)
      self.intermediate_rules[layer_idx] = {'final_rules': rules, 'raw_rules': rules_features}
      return rules_features
    else:
      rules_features = self.rule_extractor[RuleExtractorEnum.ONERULE].extract_rules(X_xai, y_rule)
    self.intermediate_rules[layer_idx] = {'final_rules': rules, 'raw_rules': rules_features}
    return rules_features

  def extract_rules(self, 
                    X: np.array, 
                    y: np.array, 
                    sample: float = None, 
                    layer_idx: List[int]= None) -> AbstractRuleSet:
    """Extract the rule set to explain the full model.

    :param X: Input features array
    :type X: np.array
    :param y: predicted values
    :type y: np.array
    :param sample: If not None percentage to sample from the training set, defaults to None
    :type sample: float, optional
    :param layer_idx: list of layers to consider into the rule extraction, defaults to None
    :type layer_idx: List[int], optional
    :raises Exception: The model is not Sequential of Functional Tensorflow model.
    :raises Exception: Layer index is not a list of valid numerical indexes.
    :raises Exception: Layer index is out of bounds of the model layer list.
    :raises Exception: Layer index is out of bounds of the model layer list.
    :return: Rule set with model global explanations.
    :rtype: AbstractRuleSet
    """
    # check model instance 
    model_type = None 
    if isinstance(self.model, tf.keras.Sequential):
      model_type = "sequential"
    elif isinstance(self.model, tf.keras.Model):
      model_type = "functional"
    else:
      raise Exception(f"The model is not sequential or functional API and cannot be processed.")
    if layer_idx is None:
      layers = self.model.layers 
      start_value = 0
      if model_type == "functional":
        start_value = 1
      # detect candidate layers  from functional model
      candidate_layers = []
      for layer_idx in range(start_value, len(layers)-1):
        if isinstance(layers[layer_idx], tf.keras.layers.Dense):
          candidate_layers.append(layer_idx)
    else:
      if isinstance(layer_idx, list):
        candidate_layers = layer_idx
      elif isinstance(layer_idx, int):
        candidate_layers = [layer_idx]
      else:
        raise Exception(f"layer_idx must be a list of integers or a single integer.")
      model_layers = self.model.layers
      for idx in candidate_layers:
        if idx >= 0:
          if idx >= len(model_layers)-1:
            raise Exception(f"Index: {idx} is out of bounds of the model")
        elif idx < 0:
          if idx < -len(model_layers):
            raise Exception(f"Index: {idx} is out of bounds of the model")
    # extract rules from each layer 
    partial_rule_sets = []
    for layer_idx in candidate_layers:
      # print(f"Extracting rules from layer: {layer_idx}")
      partial_rule_sets.append(self.extract_rules_at_layer(X=X, y=y, layer_idx=layer_idx, sample=sample))
    # total rules 
    total_rules = []
    for rs in partial_rule_sets:
      total_rules += rs.get_rules()
    print(f"total rules: {total_rules}")
    # remove duplicate rules 
    total_rules = list(set(total_rules))
    frs = RuleSet()
    frs.add_rules(total_rules)
    self.final_rule_set = frs
    return frs