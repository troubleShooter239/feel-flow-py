from numpy import ndarray, float64, sqrt, sum, multiply, array, matmul, transpose


def find_cosine(source: ndarray | list, test: ndarray | list) -> float64:
    """Compute the cosine similarity between two vectors.

    Args:
        source (Union[np.ndarray, list]): Source vector.
        test (Union[np.ndarray, list]): Test vector.

    Returns:
        np.float64: Cosine similarity between the two vectors."""
    if isinstance(source, list):
        source = array(source)
    if isinstance(test, list):
        test = array(test)
    return 1 - (matmul(transpose(source), test) / (sqrt(sum(multiply(source, source))) * sqrt(sum(multiply(test, test)))))


def find_euclidean(source: ndarray | list, test: ndarray | list) -> float64:
    """Compute the Euclidean distance between two vectors.

    Args:
        source (Union[np.ndarray, list]): Source vector.
        test (Union[np.ndarray, list]): Test vector.

    Returns:
        np.float64: Euclidean distance between the two vectors."""
    if isinstance(source, list):
        source = array(source)
    if isinstance(test, list):
        test = array(test)
    euclidean_distance = source - test
    return sqrt(sum(multiply(euclidean_distance, euclidean_distance)))
