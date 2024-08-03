from cv2 import cvtColor, COLOR_BGR2GRAY, resize
from numpy import ndarray, sum, array, float64, expand_dims
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Conv2D, Convolution2D, Dense, Dropout, 
    Flatten, MaxPooling2D
)
from tensorflow.keras.models import Model, Sequential

from ..base.base_models import AttributeModelBase
from ..commons.settings import models_settings
from .recognition_clients import VggFaceClient


class ApparentAgeClient(AttributeModelBase):
    def __init__(self) -> None:
        """Initialize the ApparentAgeClient."""
        self.model = self.load_model()

    def predict(self, img: ndarray) -> float64:
        """Predict the apparent age from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.float64: Predicted apparent age."""
        return sum(self.model.predict(img, verbose=0)[0, :] * array(list(range(0, 101))))

    @classmethod
    def load_model(cls) -> Model:
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
        age_model.load_weights(cls._download(models_settings.face_attrs.age))
        return age_model


class EmotionClient(AttributeModelBase):
    labels: tuple[str, str, str, str, str, str, str] = (
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    )

    def __init__(self):
        """Initialize the EmotionClient."""
        self.model = self.load_model()

    def predict(self, img: ndarray) -> ndarray:
        """Predict the emotion from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted emotion probabilities."""
        img_gray = cvtColor(img[0], COLOR_BGR2GRAY)
        img_gray = resize(img_gray, (48, 48))
        return self.model.predict(expand_dims(img_gray, axis=0), verbose=0)[0, :]

    @classmethod
    def load_model(cls) -> Sequential:
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
        model.load_weights(cls._download(models_settings.face_attrs.emotion))
        return model


class GenderClient(AttributeModelBase):
    labels: tuple[str, str] = ("woman", "man")

    def __init__(self):
        """Initialize the GenderClient."""
        self.model = self.load_model()

    def predict(self, img: ndarray) -> ndarray:
        """Predict the gender from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted gender probabilities."""
        return self.model.predict(img, verbose=0)[0, :]

    @classmethod
    def load_model(cls) -> Model:
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
        gender_model.load_weights(cls._download(models_settings.face_attrs.gender))
        return gender_model


class RaceClient(AttributeModelBase):
    labels: tuple[str, str, str, str, str, str] = (
        "asian", "indian", "black", "white", "middle_eastern", "latino_hispanic"
    )

    def __init__(self):
        """Initialize the RaceClient."""
        self.model = self.load_model()

    def predict(self, img: ndarray) -> ndarray:
        """Predict the race from the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted race probabilities."""
        return self.model.predict(img, verbose=0)[0, :]

    @classmethod
    def load_model(cls) -> Model:
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
        race_model.load_weights(cls._download(models_settings.face_attrs.race))
        return race_model
