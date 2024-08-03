from base64 import b64decode

from cv2 import imdecode, IMREAD_COLOR, imread
from numpy import ndarray, fromstring, uint8, array, frombuffer
from pathlib import Path
from PIL import Image
from requests import get


def load_base64(uri: str) -> ndarray:
    """Loads an image from a base64-encoded string.

    Args:
        uri (str): Base64-encoded image string.

    Returns:
        np.ndarray: Loaded image as a NumPy array."""
    return imdecode(fromstring(b64decode(uri.split(",")[1]), uint8), IMREAD_COLOR)


def load_image(img: bytes | str | ndarray) -> tuple[ndarray, str]:
    """Loads an image from various sources.

    Args:
        img bytes | str | ndarray): Input image, either as a file path, URL, base64-encoded string, or NumPy array.

    Returns:
        tuple[ndarray, str]: Loaded image as a NumPy array and the source type."""
    if isinstance(img, bytes):
        return frombuffer(img, dtype=uint8), "bytes"
    if isinstance(img, ndarray):
        return img, "numpy array"
    if isinstance(img, Path):
        img = str(img)
    if isinstance(img, str):
        if img.startswith("data:image/"):
            return load_base64(img), "base64 encoded string"
        if img.startswith("http"):
            return (array(Image.open(get(img, stream=True, timeout=60).raw).convert("BGR")), img)
        return imread(img), img
    raise ValueError("Unsupported image format or type.")