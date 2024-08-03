from abc import ABC, abstractmethod
from os import makedirs
from os.path import isfile, exists

from requests import get
from numpy import ndarray, float64
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Activation, add, BatchNormalization, Concatenate, Conv2D, Dense, 
    Dropout, Input, GlobalAveragePooling2D, Lambda, MaxPooling2D
)
from tensorflow.keras.models import Model

from ..commons.settings import models_settings


class BaseModel(ABC):
    """Abstract base class for deep learning models."""
    def __init__(self) -> None:
        self.model: Model
        self.model_name: str

    @staticmethod
    def _download(filename: str) -> str:
        if not exists(models_settings.download_folder):
            makedirs(models_settings.download_folder, exist_ok=True)

        file_path = models_settings.download_folder + "/" + filename
        if not isfile(file_path):
            with get(models_settings.download_url + filename, stream=True) as r:
                print(f"Trying http-get: " + r.url)
                r.raise_for_status()
                print(f"Started downloading file: " + filename)
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"Download completed!")
        return file_path


class AttributeModelBase(BaseModel):
    """Abstract base class for attribute prediction models."""

    @abstractmethod
    def predict(self, img: ndarray) -> ndarray | float64: 
        """Predict attributes for a given image.

        Args:
            img (ndarray): Input image.

        Returns:
            Union[ndarray, float64]: Predicted attributes."""
        pass


class FacialRecognitionBase(BaseModel):
    """Abstract base class for facial recognition models."""

    @abstractmethod 
    def find_embeddings(self, img: ndarray) -> list[float]: 
        """Find facial embeddings for a given image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        pass


class FaceNetBase(FacialRecognitionBase):
    """Base class for the FaceNet facial recognition model."""

    @staticmethod
    def _inception_res_netV2(dimension: int = 128) -> Model:
        """Create an Inception-ResNetV2 model for FaceNet.

        Args:
            dimension (int): Embedding dimension. Defaults to 128.

        Returns:
            Model: Inception-ResNetV2 model."""
        def scaling(x, scale): 
            return x * scale

        inputs = Input(shape=(160, 160, 3))
        
        x = Conv2D(32, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_1a_3x3")(inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_1a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_1a_3x3_Activation")(x)
        x = Conv2D(32, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_2a_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_2a_3x3_Activation")(x)
        x = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="Conv2d_2b_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2b_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_2b_3x3_Activation")(x)
        x = MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3")(x)
        x = Conv2D(80, 1, strides=1, padding="valid", use_bias=False, name="Conv2d_3b_1x1")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_3b_1x1_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_3b_1x1_Activation")(x)
        x = Conv2D(192, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_4a_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_4a_3x3_Activation")(x)
        x = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_4b_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4b_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_4b_3x3_Activation")(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
       
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_1_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        
        x = add([x, up])
        x = Activation("relu", name="Block35_1_Activation")(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm",)(branch_1)
        branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_2_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_2_Activation")(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False,name="Block35_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
       
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
       
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_3_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_3_Activation")(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
       
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
       
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_4_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_4_Activation")(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_5_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_5_Activation")(x)
        
        branch_0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_0_Conv2d_1a_3x3")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
        
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_1a_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
        
        branch_pool = MaxPooling2D(3, strides=2, padding="valid", name="Mixed_6a_Branch_2_MaxPool_1a_3x3")(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name="Mixed_6a")(branches)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_1_Branch_1_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_1_Branch_1_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_1_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_1_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_2_Branch_2_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_2_Branch_2_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_2_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_2_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_3_Branch_3_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_3_Branch_3_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_3_Branch_3_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_3_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_3_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_4_Branch_4_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_4_Branch_4_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_4_Branch_4_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_4_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_4_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_5_Branch_5_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_5_Branch_5_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False,  name="Block17_5_Branch_5_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0c_7x1_Activation")(branch_1)
       
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_5_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_5_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_6_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_6_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
       
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_6_Branch_6_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_6_Branch_6_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,  name="Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_6_Branch_6_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm",)(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_6_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_6_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_6_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_7_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_7_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_7_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_7_Branch_7_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_7_Branch_7_Conv2d_0b_1x7",)(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_7_Branch_7_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_7_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_7_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_7_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_8_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_8_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_8_Branch_0_Conv2d_1x1_Activation")(branch_0)
       
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_8_Branch_8_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_8_Branch_8_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_8_Branch_8_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_8_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_8_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_8_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_9_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_9_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_9_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_9_Branch_9_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_9_Branch_9_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_9_Branch_9_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,  name="Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_9_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_9_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_9_Activation")(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_10_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_10_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_10_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name="Block17_10_Branch_10_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name="Block17_10_Branch_10_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name="Block17_10_Branch_10_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0c_7x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_10_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_10_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_10_Activation")(x)
        
        branch_0 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_0a_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation")(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_1a_3x3")(branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
       
        branch_1 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_1a_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
        
        branch_2 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_1a_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation")(branch_2)
        
        branch_pool = MaxPooling2D(3, strides=2, padding="valid", name="Mixed_7a_Branch_3_MaxPool_1a_3x3")(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name="Mixed_7a")(branches)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_1_Branch_1_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_1_Branch_1_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_1_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_1_Activation")(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_2_Branch_2_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_2_Branch_2_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_2_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_2_Activation")(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
       
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_3_Branch_3_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_3_Branch_3_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_3_Branch_3_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_3_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_3_Activation")(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_4_Branch_4_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_4_Branch_4_Conv2d_0b_1x3",)(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_4_Branch_4_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_4_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_4_Activation")(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_5_Branch_5_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_5_Branch_5_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_5_Branch_5_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_5_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_5_Activation")(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
        
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_6_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_6_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 1})(up)
        x = add([x, up])
        x = GlobalAveragePooling2D(name="AvgPool")(x)
        x = Dropout(1.0 - 0.8, name="Dropout")(x)
        x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm")(x)
        
        return Model(inputs, x, name="inception_resnet_v1")

    @classmethod
    def load_model(cls) -> Model:
        """Load the FaceNet model.

        Args:
            url (str): URL from which to download the model weights. Defaults to C.DOWNLOAD_URL_FACENET.

        Returns:
            Model: Loaded FaceNet model."""
        if cls.__class__.__name__ == "FaceNet128dClient":
            model = FaceNetBase._inception_res_netV2()
            filename = models_settings.recognition_attrs.facenet
        else:
            model = FaceNetBase._inception_res_netV2(512)
            filename = models_settings.recognition_attrs.facenet512
        model.load_weights(cls._download(filename))
        return model
