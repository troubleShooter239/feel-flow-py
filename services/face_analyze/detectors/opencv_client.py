from os.path import isfile, sep
from typing import Any

import cv2

from base.base_detector import DetectorBase, ndarray, List, Tuple


class OpenCvClient(DetectorBase):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict:
        return {
            "face_detector": self.__build_cascade("haarcascade"),
            "eye_detector": self.__build_cascade("haarcascade_eye")
        }

    def detect_faces(self, img: ndarray, 
                     align: bool = True) -> List[Tuple[ndarray, List[float], float]]:
        img_region = [0, 0, img.shape[1], img.shape[0]]
        faces = []
        
        try:
            faces, _, scores = self.model["face_detector"].detectMultiScale3(
                img, 1.1, 10, outputRejectLevels=True
            )
        except:
            pass

        if len(faces) == 0:
            return []

        resp = []
        for (x, y, w, h), confidence in zip(faces, scores):
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

            if align:
                left_eye, right_eye = self.find_eyes(detected_face)
                detected_face = self._align_face(detected_face, left_eye, right_eye)

            img_region = [x, y, w, h]

            resp.append((detected_face, img_region, confidence))

        return resp

    def find_eyes(self, img: ndarray) -> tuple:
        left_eye = None
        right_eye = None

        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye

        eyes = self.model["eye_detector"].detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 10)
        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        if len(eyes) >= 2:
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            left_eye = (int(left_eye[0] + (left_eye[2] / 2)), 
                        int(left_eye[1] + (left_eye[3] / 2)))
            right_eye = (int(right_eye[0] + (right_eye[2] / 2)),
                         int(right_eye[1] + (right_eye[3] / 2)))
            
        return left_eye, right_eye

    def __build_cascade(self, model_name="haarcascade") -> Any:
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
            if not isfile(face_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path, " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = opencv_path + "haarcascade_eye.xml"
            if not isfile(eye_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated."
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def __get_opencv_path(self) -> str:
        folders = cv2.__file__.split(sep)[0:-1]
        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        return path + "/data/"


class DetectorWrapper:
    @staticmethod
    def build_model() -> OpenCvClient:
        global face_detector_obj
        if not "face_detector_obj" in globals():
            face_detector_obj = {}
        detector_backend = "opencv"
        built_models = list(face_detector_obj.keys())
        if detector_backend not in built_models:
            face_detector_obj[detector_backend] = OpenCvClient()            
        return face_detector_obj[detector_backend]

    @staticmethod
    def detect_faces(img: ndarray, align: bool = True) -> list:
        return DetectorWrapper.build_model().detect_faces(img=img, align=align)
