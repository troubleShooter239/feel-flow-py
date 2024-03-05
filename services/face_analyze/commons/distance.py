from typing import Union

import numpy as np
from numba import njit


def find_cosine(source: Union[np.ndarray, list], test: Union[np.ndarray, list]) -> np.float64:
    if isinstance(source, list):
        source = np.array(source)
    if isinstance(test, list):
        test = np.array(test)
    return 1 - (np.matmul(np.transpose(source), test) / (np.sqrt(np.sum(np.multiply(source, source))) * np.sqrt(np.sum(np.multiply(test, test)))))

@njit
def find_euclidean(source: Union[np.ndarray, list], test: Union[np.ndarray, list]) -> np.float64:
    if isinstance(source, list):
        source = np.array(source)
    if isinstance(test, list):
        test = np.array(test)
    euclidean_distance = source - test
    return np.sqrt(np.sum(np.multiply(euclidean_distance, euclidean_distance)))
