import os
from pathlib import Path

from .constants import _DEEPFACE_HOME


def get_deepface_home() -> str:
    return str(os.getenv(_DEEPFACE_HOME, str(Path.home())))


def initialize_folder() -> None:
    deepFaceHomePath = get_deepface_home() + "/.deepface"
    if not os.path.exists(deepFaceHomePath):
        os.makedirs(deepFaceHomePath, exist_ok=True)
    weightsPath = deepFaceHomePath + "/weights"
    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)