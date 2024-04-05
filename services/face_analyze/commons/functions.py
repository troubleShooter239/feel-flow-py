from typing import Any, Dict, List, Tuple, Union

import numpy as np
from cv2 import COLOR_BGR2GRAY, resize, cvtColor
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from numba import njit

from ..models import face_attributes as fa
from ..models import recognition_models as rm
from ..detectors.opencv_client import DetectorWrapper
from ..loaders.image_loader import load_image


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
        "VGG-Face": rm.VggFaceClient,
        "OpenFace": rm.OpenFaceClient,
        "Facenet": rm.FaceNet128dClient,
        "Facenet512": rm.FaceNet512dClient,
        "DeepFace": rm.DeepFaceClient,
        "DeepID": rm.DeepIdClient,
        "Dlib": rm.DlibClient,
        "ArcFace": rm.ArcFaceClient,
        "SFace": rm.SFaceClient,
        "Emotion": fa.EmotionClient,
        "Age": fa.ApparentAgeClient,
        "Gender": fa.GenderClient,
        "Race": fa.RaceClient
    }
    if not model_name in model_obj:
        model_obj[model_name] = models[model_name]()
    return model_obj[model_name]


@njit
def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """Normalizes the input vector.

    Args:
        x (Union[np.ndarray, list]): Input vector.

    Returns:
        np.ndarray: Normalized vector."""
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_threshold(model_name: str, distance_metric: str) -> float:
    """Finds the threshold value based on the model and distance metric.

    Args:
        model_name (str): Name of the recognition model.
        distance_metric (str): Distance metric used for comparison.

    Returns:
        float: Threshold value."""
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}
    thresholds = {
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17}
    }
    return thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)



def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
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
    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
    elif normalization == "Facenet2018":
        img = img / 127.5 - 1
    elif normalization == "VGGFace":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
    elif normalization == "VGGFace2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
    elif normalization == "ArcFace":
        img = (img - 127.5) / 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")
    return img


def find_size(model_name: str) -> Tuple[int, int]:
    """Finds the image size based on the recognition model.

    Args:
        model_name (str): Name of the recognition model.

    Returns:
        Tuple[int, int]: Image size."""
    sizes = {
        "VGG-Face": (224, 224),
        "Facenet": (160, 160),
        "Facenet512": (160, 160),
        "OpenFace": (96, 96),
        "DeepFace": (152, 152),
        "DeepID": (47, 55),
        "Dlib": (150, 150),
        "ArcFace": (112, 112),
        "SFace": (112, 112),
    }
    try:
        return sizes[model_name]
    except KeyError:
        return (0, 0)


def represent(img_path: Union[str, np.ndarray], model_name: str = "VGG-Face", 
              enforce_detection: bool = True, detector_backend: str = "opencv",
              align: bool = True, normalization: str = "base") -> List[Dict[str, Any]]:
    """Represent an image with the specified recognition model.

    Args:
        img_path (Union[str, np.ndarray]): Path to the image or image array.
        model_name (str, optional): Name of the recognition model. Defaults to "VGG-Face".
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
        detector_backend (str, optional): Backend for face detection. Defaults to "opencv".
        align (bool, optional): Whether to align faces. Defaults to True.
        normalization (str, optional): Type of normalization. Defaults to "base".

    Returns:
        List[Dict[str, Any]]: List of representations."""
    model = build_model(model_name)
    target_size = find_size(model_name)
    if detector_backend != "skip":
        img_objs = extract_faces(img_path, (target_size[1], target_size[0]), 
                                 detector_backend, False, enforce_detection, align=align)
    else:
        img, _ = load_image(img_path)
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            img = resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)

        img_objs = [(img, {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}, 0)]
    
    resp_objs = []
    for i, r, c in img_objs:
        e = model.find_embeddings(normalize_input(i, normalization))
        resp_obj = {"embedding": e, "facial_area": r, "face_confidence": c}
        resp_objs.append(resp_obj)

    return resp_objs


def extract_faces(img: Union[str, np.ndarray], target_size: tuple = (224, 224), 
                  grayscale: bool = False, enforce_detection: bool = True, 
                  align: bool = True) -> List[Tuple[np.ndarray, Dict[str, int], float]]:
    """Extract faces from an image.

    Args:
        img (Union[str, np.ndarray]): Path to the image or image array.
        target_size (tuple, optional): Target size of the face. Defaults to (224, 224).
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to False.
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
        align (bool, optional): Whether to align faces. Defaults to True.

    Returns:
        List[Tuple[np.ndarray, Dict[str, int], float]]: List of tuples containing face images, 
        facial area coordinates, and confidence scores."""
    img, img_name = load_image(img)

    face_objs = DetectorWrapper.detect_faces(img, align)

    if len(face_objs) == 0 and enforce_detection:
        raise ValueError(f"Face could not be detected in {None if img_name is None else img_name}."
                            "Please confirm that the picture is a face photo "
                            "or consider to set enforce_detection param to False.")
    
    img_region = [0, 0, img.shape[1], img.shape[0]]
    if len(face_objs) == 0 and not enforce_detection:
        face_objs = [(img, img_region, 0)]

    extracted_faces = []
    for current_img, reg, confidence in face_objs:
        if current_img.shape[0] < 0 or current_img.shape[1] < 0:
            continue

        if grayscale:
            current_img = cvtColor(current_img, COLOR_BGR2GRAY)

        factor = min(target_size[0] / current_img.shape[0], 
                     target_size[1] / current_img.shape[1])

        current_img = resize(current_img, 
                                (int(current_img.shape[1] * factor), 
                                int(current_img.shape[0] * factor)))

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]
        
        if grayscale:
            current_img = np.pad(current_img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2)), "constant")
        else:
            current_img = np.pad(current_img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant")

        if current_img.shape[0:2] != target_size:
            current_img = resize(current_img, target_size)

        img_pixels = np.expand_dims(image.img_to_array(current_img), axis=0) / 255
        regs = {"x": int(reg[0]), "y": int(reg[1]), "w": int(reg[2]), "h": int(reg[3])}
        extracted_faces.append((img_pixels, regs, confidence))

    if len(extracted_faces) == 0 and enforce_detection:
        raise ValueError(f"Detected face shape is {img.shape}. "
                         "Consider to set enforce_detection arg to False.")

    return extracted_faces
