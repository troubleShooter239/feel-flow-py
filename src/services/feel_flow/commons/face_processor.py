from numpy import argmax

    # return VerifyResult(verified=True if distance <= threshold else False, distance=float(distance), 
    #                     threshold=threshold, model=model_name, distance_metric=distance_metric, 
    #                     facial_areas=FacialAreas(img1=Region(**facial_areas[0]), 
    #                                              img2=Region(**facial_areas[1])))

from services.feel_flow.models.response_models import Emotion, Gender, Race

from ..clients.face_clients import EmotionClient, GenderClient, RaceClient

class FaceProcessor:
    @staticmethod
    def age(predictions) -> dict[str, int]:
        """Process age predictions.

        Args:
            predictions: Predicted age.

        Returns:
            Dict[str, int]: Processed age prediction."""
        return {"age": int(predictions)}

    @staticmethod
    def emotion(predictions) -> dict[str, Emotion | str]:
        """Process emotion predictions.

        Args:
            predictions: Predicted emotion probabilities.

        Returns:
            Dict[str, Union[Dict[str, float], str]]: Processed emotion predictions."""
        return {
            "emotion": Emotion(**{l: round(100 * p / predictions.sum(), 2) for l, p in zip(EmotionClient.labels, predictions)}),
            "dominant_emotion": EmotionClient.labels[argmax(predictions)]
        }

    @staticmethod
    def gender(predictions) -> dict[str, Gender | str]:
        """Process gender predictions.

        Args:
            predictions: Predicted gender probabilities.

        Returns:
            Dict[str, Union[Dict[str, float], str]]: Processed gender predictions."""
        return {
            "gender": Gender(**{l: round(100 * p, 2) for l, p in zip(GenderClient.labels, predictions)}),
            "dominant_gender": GenderClient.labels[argmax(predictions)]
        }         

    @staticmethod
    def race(predictions) -> dict[str, Race | str]:
        """Process race predictions.

        Args:
            predictions: Predicted race probabilities.

        Returns:
            Dict[str, Union[Dict[str, float], str]]: Processed race predictions."""
        return {
            "race": Race(**{l: round(100 * p / predictions.sum(), 2) for l, p in zip(RaceClient.labels, predictions)}),
            "dominant_race": RaceClient.labels[argmax(predictions)]
        }
