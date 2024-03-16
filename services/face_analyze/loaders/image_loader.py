from base64 import b64decode
from typing import Tuple, Union

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from requests import get


def load_base64(uri: str) -> np.ndarray:
    """Loads an image from a base64-encoded string.

    Args:
        uri (str): Base64-encoded image string.

    Returns:
        np.ndarray: Loaded image as a NumPy array."""
    return cv2.imdecode(np.fromstring(b64decode(uri.split(",")[1]), np.uint8), 
                        cv2.IMREAD_COLOR)


def load_image(img: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """Loads an image from various sources.

    Args:
        img (Union[str, np.ndarray]): Input image, either as a file path, URL, base64-encoded string, or NumPy array.

    Returns:
        Tuple[np.ndarray, str]: Loaded image as a NumPy array and the source type."""
    if isinstance(img, np.ndarray):
        return img, "numpy array"
    if isinstance(img, Path):
        img = str(img)
    if img.startswith("data:image/"):
        return load_base64(img), "base64 encoded string"
    if img.startswith("http"):
        return (np.array(Image.open(get(img, stream=True, 
                                        timeout=60).raw).convert("BGR")), img)
    return cv2.imread(img), img
