import os
from pathlib import Path

from .constants import _DEEPFACE_HOME


def get_deepface_home() -> str:
    """Get the path to the DeepFace home directory.

    Returns:
        str: Path to the DeepFace home directory."""
    return str(os.getenv(_DEEPFACE_HOME, str(Path.home())))


def initialize_folder() -> None:
    """Initialize DeepFace folders.

    This function creates necessary folders for DeepFace if they do not exist."""
    deepFaceHomePath = get_deepface_home() + "/.deepface"
    if not os.path.exists(deepFaceHomePath):
        os.makedirs(deepFaceHomePath, exist_ok=True)
    weightsPath = deepFaceHomePath + "/weights"
    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)