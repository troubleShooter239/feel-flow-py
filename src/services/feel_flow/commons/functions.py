from numpy import ndarray, sum, array, sqrt, multiply
from tensorflow.keras.models import Model

from ..clients import face_clients as fc
from ..clients import recognition_clients as rc


def build_model(model_name: str) -> Model:
    """Builds and returns the specified recognition model.

    Args:
        model_name (str): Name of the recognition model.

    Returns:
        Model: Instance of the specified recognition model."""
    global model_obj
    if not "model_obj" in globals():
        model_obj = dict()
    models = {
        "vgg_face": rc.VggFaceClient,
        "openface": rc.OpenFaceClient,
        "facenet": rc.FaceNet128dClient,
        "facenet512": rc.FaceNet512dClient,
        "deepid": rc.DeepIdClient,
        "arcface": rc.ArcFaceClient,
        "age": fc.ApparentAgeClient,
        "emotion": fc.EmotionClient,
        "gender": fc.GenderClient,
        "race": fc.RaceClient
    }
    if not model_name in model_obj:
        model_obj[model_name] = models[model_name]()
    return model_obj[model_name]


def l2_normalize(x: ndarray | list) -> ndarray:
    """Normalizes the input vector.

    Args:
        x (Union[np.ndarray, list]): Input vector.

    Returns:
        np.ndarray: Normalized vector."""
    return array(x) / sqrt(sum(multiply(x, x))) if isinstance(x, list) else x / sqrt(sum(multiply(x, x)))


def find_threshold(model_name: str, distance_metric: str) -> float:
    """Finds the threshold value based on the model and distance metric.

    Args:
        model_name (str): Name of the recognition model.
        distance_metric (str): Distance metric used for comparison.

    Returns:
        float: Threshold value."""
    return {
        "vgg_face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},
        "facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "arcface": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "ppenface": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "deepid": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17}
    }.get(model_name, {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}).get(distance_metric, 0.4)


def normalize_input(img: ndarray, normalization: str = "base") -> ndarray:
    """Normalize input image.

    Args:
        img (np.ndarray): Input image.
        normalization (str, optional): Type of normalization. Defaults to "base".

    Returns:
        np.ndarray: Normalized image."""
    if normalization == "base":
        return img
    img *= 255
    if normalization == "raw":
        pass
    elif normalization == "facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
    elif normalization == "facenet512":
        img = img / 127.5 - 1
    elif normalization == "vggface":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
    elif normalization == "vggface2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
    elif normalization == "arcface":
        img = (img - 127.5) / 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")
    return img


def find_size(model_name: str) -> tuple[int, int]:
    """Finds the image size based on the recognition model.

    Args:
        model_name (str): Name of the recognition model.

    Returns:
        Tuple[int, int]: Image size."""
    sizes = {
        "vgg_face": (224, 224),
        "facenet": (160, 160),
        "facenet512": (160, 160),
        "openface": (96, 96),
        "deepid": (47, 55),
        "arcface": (112, 112),
    }
    try:
        return sizes[model_name]
    except KeyError:
        return (0, 0)
