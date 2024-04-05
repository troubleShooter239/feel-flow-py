import cv2
import numpy as np
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Conv2D, Convolution2D, Dense, 
    Dropout, Flatten, MaxPooling2D
)
from tensorflow.keras.models import Model, Sequential

from ..base.base_models import AttributeModelBase
from ..commons import constants as C
from ..commons.folder_utils import get_deepface_home
from .recognition_models import VggFaceClient


class ApparentAgeClient(AttributeModelBase):
    def __init__(self) -> None:
        """Initialize the ApparentAgeClient."""
        self.model, self.model_name = self.load_model(), "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        """Predict the apparent age from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.float64: Predicted apparent age."""
        return np.sum(self.model.predict(img, verbose=0)[0, :] * np.array(list(range(0, 101))))

    def load_model(self, url: str = C.DOWNLOAD_URL_AGE) -> Model:
        """Load the model for apparent age prediction.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_AGE.

        Returns:
            Model: Loaded model."""
        model = VggFaceClient.base_model()
        base_out = Sequential()
        base_out = Convolution2D(101, (1, 1), name="predictions")(model.layers[-4].output)
        base_out = Flatten()(base_out)
        base_out = Activation("softmax")(base_out)
        age_model = Model(inputs=model.input, outputs=base_out)
        output = get_deepface_home() + C.PATH_WEIGHTS_AGE
        self._download(url, output)
        age_model.load_weights(output)
        return age_model


class EmotionClient(AttributeModelBase):
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        """Initialize the EmotionClient."""
        self.model, self.model_name = self.load_model(), "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict the emotion from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted emotion probabilities."""
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)
        return self.model.predict(img_gray, verbose=0)[0, :]

    def load_model(self, url: str = C.DOWNLOAD_URL_EMOTION) -> Sequential:
        """Load the model for emotion prediction.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_EMOTION.

        Returns:
            Sequential: Loaded model."""
        num_classes = 7
        model = Sequential()
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation="softmax"))
        output = get_deepface_home() + C.PATH_WEIGHTS_EMOTION
        self._download(url, output)
        model.load_weights(output)
        return model


class GenderClient(AttributeModelBase):
    labels = ["woman", "man"]

    def __init__(self):
        """Initialize the GenderClient."""
        self.model, self.model_name = self.load_model(), "Gender"

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict the gender from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted gender probabilities."""
        return self.model.predict(img, verbose=0)[0, :]

    def load_model(self, url: str = C.DOWNLOAD_URL_GENDER) -> Model:
        """Load the model for gender prediction.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_GENDER.

        Returns:
            Model: Loaded model."""
        model = VggFaceClient.base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(2, (1, 1), name="predictions")(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        output = get_deepface_home() + C.PATH_WEIGHTS_GENDER
        self._download(url, output)
        gender_model.load_weights(output)
        return gender_model


class RaceClient(AttributeModelBase):
    labels = ["asian", "indian", "black", "white", "middle_eastern", "latino_hispanic"]

    def __init__(self):
        """Initialize the RaceClient."""
        self.model, self.model_name = self.load_model(), "Race"

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict the race from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted race probabilities."""
        return self.model.predict(img, verbose=0)[0, :]

    def load_model(self, url: str = C.DOWNLOAD_URL_RACE) -> Model:
        """Load the model for race prediction.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_RACE.

        Returns:
            Model: Loaded model."""
        model = VggFaceClient.base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(6, (1, 1), name="predictions")(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        race_model = Model(inputs=model.input, outputs=base_model_output)
        output = get_deepface_home() + C.PATH_WEIGHTS_RACE
        self._download(url, output)
        race_model.load_weights(output)
        return race_model
