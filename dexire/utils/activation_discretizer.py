
import numpy as np
from typing import List, Any, Tuple

def digitalize_row(
    x: np.ndarray,
    bins:List[Any]=None, 
    n_bins:int=2) -> np.ndarray:
    """Digitalize one column of the input array.

    :param X: Numpy array with the input features.
    :type X: np.ndarray
    :param col_idx: Column to be discretized in X. 
    :type col_idx: int
    :param bins: List of bins to discretized column identified by col_idx, defaults to None
    :type bins: List[Any], optional
    :param n_bins: Bins number, defaults to 10
    :type n_bins: int, optional
    :return: discretized column and bins. 
    :rtype: Tuple[np.ndarray, List[Any]]
    """
    temp_x = x
    if bins is None:
        if n_bins == 2:
            threshold = np.mean(temp_x)
            return np.where(temp_x >= threshold, 1, 0)
        max_val = np.max(temp_x)
        min_val = np.min(temp_x)
        bins = np.linspace(min_val, max_val, n_bins)
    digitalized_column = np.digitize(temp_x, bins)
    return digitalized_column

def discretize_activation_layer(X: np.ndarray,
                                bins: List[Any] = None,
                                n_bins: int = 2) -> np.ndarray:
    """Discretize the activation layer with numpy.

    :param X: Activations matrix to be discretized.
    :type X: np.ndarray
    :param bins: List of bins to discretize, defaults to None
    :type bins: List[Any], optional
    :param n_bins: number of bins to discretize data, defaults to 2
    :type n_bins: int, optional
    :raises Exception: The X array has incorrect shape.
    :return: Discretized activation layer.
    :rtype: np.ndarray
    """
    if X.ndim == 2:
        if bins is None and n_bins == 2:
            thresholds = np.mean(X, axis=0, keepdims=True)
            return (X >= thresholds).astype(int)
        axis_application = np.apply_along_axis(
            digitalize_row,
            axis=0,
            arr=X,
            bins=bins,
            n_bins=n_bins
        )
        return axis_application
    else:
        raise Exception(f"The array to discretize should be rank 2 and is {X.ndim}")