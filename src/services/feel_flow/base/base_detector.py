from abc import ABC, abstractmethod

from PIL.Image import fromarray
from cv2 import CascadeClassifier
from numpy import arctan2, array, degrees, ndarray


class DetectorBase(ABC):
    @staticmethod
    def _align_face(img: ndarray, left_eye: tuple[int, int], right_eye: tuple[int, int]) -> ndarray:
        """Aligns the face in the image based on the position of the eyes.

        Args:
            img (ndarray): Input image.
            left_eye (Union[list, tuple]): Coordinates of the left eye.
            right_eye (Union[list, tuple]): Coordinates of the right eye.

        Returns:
            ndarray: Aligned image."""
        if (left_eye is None or right_eye is None) or (img.shape[0] == 0 or img.shape[1] == 0):
            return img
        
        return array(fromarray(img).rotate(float(
            degrees(arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        )))
    
    @abstractmethod
    def build_model(self) -> dict[str, CascadeClassifier]:
        """Builds and returns the model for face detection.

        Returns:
            dict: Model parameters or configuration."""
        pass

    @abstractmethod
    def detect_faces(self, img: ndarray, align: bool = True) -> tuple[tuple[ndarray, tuple[int, int, int, int], float]]: 
        """Detects faces in the input image.

        Args:
            img (ndarray): Input image.
            align (bool): Whether to align detected faces. Defaults to True.

        Returns:
            List[Tuple[ndarray, List[float], float]]: A list of tuples containing detected face images, 
                face landmarks, and confidence scores."""
        pass
