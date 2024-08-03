from os.path import isfile, sep

from cv2 import cvtColor, COLOR_BGR2GRAY, CascadeClassifier

from ..base.base_detector import DetectorBase, ndarray


class OpenCvClient(DetectorBase):
    """Class for face detection using OpenCV.

    Attributes:
        model (dict): Dictionary containing the built face and eye detector models."""
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict[str, CascadeClassifier]:
        """Builds and returns the face and eye detector models.

        Returns:
            dict: Dictionary containing the face and eye detector models.
        """
        return {
            "face_detector": self.__build_cascade("haarcascade"),
            "eye_detector": self.__build_cascade("haarcascade_eye")
        }

    def detect_faces(self, img: ndarray, align: bool = True) -> tuple[tuple[ndarray, tuple[int, int, int, int], float]]:
        """Detects faces in the input image.

        Args:
            img (ndarray): Input image.
            align (bool): Whether to align detected faces. Defaults to True.

        Returns:
            List[Tuple[ndarray, List[float], float]]: A list of tuples containing detected face images, 
                face region coordinates, and confidence scores."""
        faces = []
        
        try:
            faces, _, scores = self.model["face_detector"].detectMultiScale3(img, 1.1, 10, outputRejectLevels=True)
        except Exception:
            pass

        if len(faces) == 0:
            raise ValueError("no faces was found")

        resp = []
        for (x, y, w, h), confidence in zip(faces, scores):
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

            if align:
                left_eye, right_eye = self.find_eyes(detected_face)
                detected_face = self._align_face(detected_face, left_eye, right_eye)

            resp.append((detected_face, (x, y, w, h), confidence))

        return tuple(resp)

    def find_eyes(self, img: ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """Finds the coordinates of the eyes in the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            tuple: Tuple containing the coordinates of the left and right eyes."""
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("ad image shape")

        eyes = self.model["eye_detector"].detectMultiScale(cvtColor(img, COLOR_BGR2GRAY), 1.1, 10)
        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        if len(eyes) < 2:
            raise ValueError("no eyes was found")

        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
            
        return ((int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2))), 
                (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2))))

    def __build_cascade(self, model_name="haarcascade") -> CascadeClassifier:
        """Builds a cascade classifier model.

        Args:
            model_name (str, optional): Name of the model. Defaults to "haarcascade".

        Raises:
            ValueError: If the specified model name is not implemented.

        Returns:
            cv2.CascadeClassifier: Cascade classifier model."""
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
            if not isfile(face_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ", face_detector_path, " violated."
                )
            detector = CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = opencv_path + "haarcascade_eye.xml"
            if not isfile(eye_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ", eye_detector_path, " violated."
                )
            detector = CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def __get_opencv_path(self) -> str:
        """Gets the path to the OpenCV data directory.

        Returns:
            str: Path to the OpenCV data directory."""
        import cv2
        folders = cv2.__file__.split(sep)[0:-1]
        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        return path + "/data/"


class DetectorWrapper:
    """Wrapper class for the face detector.

    Provides a unified interface to build and detect faces using different backends."""
    @staticmethod
    def build_model() -> OpenCvClient:
        """Builds the face detector model.

        Returns:
            OpenCvClient: Instance of the OpenCvClient class."""
        global face_detector_obj
        if not "face_detector_obj" in globals():
            face_detector_obj = {}
        detector_backend = "opencv"
        built_models = list(face_detector_obj.keys())
        if detector_backend not in built_models:
            face_detector_obj[detector_backend] = OpenCvClient()            
        return face_detector_obj[detector_backend]

    @staticmethod
    def detect_faces(img: ndarray, align: bool = True) -> tuple[tuple[ndarray, tuple[int, int, int, int], float]]:
        """Detects faces in the input image.

        Args:
            img (ndarray): Input image.
            align (bool): Whether to align detected faces. Defaults to True.

        Returns:
            list: List of tuples containing detected face images, region coordinates, and confidence scores."""
        return DetectorWrapper.build_model().detect_faces(img=img, align=align)
