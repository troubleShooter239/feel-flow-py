from typing import Dict, Union

import numpy as np

from models.face_attributes import EmotionClient, GenderClient, RaceClient


class FaceProcessor:
    @staticmethod
    def age(predictions) -> Dict[str, int]:
        return {"age": int(predictions)}

    @staticmethod
    def emotion(predictions) -> Dict[str, Union[Dict[str, float], str]]:
        _sum = predictions.sum()
        return {
            "emotion": {l: round(100 * p / _sum, 2) for l, p in zip(EmotionClient.labels, predictions)},
            "dominant_emotion": EmotionClient.labels[np.argmax(predictions)]
        }

    @staticmethod
    def gender(predictions) -> Dict[str, Union[Dict[str, float], str]]:
        return {
            "gender": {l: round(100 * p, 2) for l, p in zip(GenderClient.labels, predictions)},
            "dominant_gender": GenderClient.labels[np.argmax(predictions)]
        }         

    @staticmethod
    def race(predictions) -> Dict[str, Union[Dict[str, float], str]]:
        _sum = predictions.sum()
        return {
            "race": {l: round(100 * p / _sum, 2) for l, p in zip(RaceClient.labels, predictions)},
            "dominant_race": RaceClient.labels[np.argmax(predictions)]
        }
