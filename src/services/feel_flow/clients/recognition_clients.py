import tensorflow as tf
from tensorflow.keras.backend import sqrt, l2_normalize
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Add, BatchNormalization, concatenate, Conv2D, 
    Convolution2D, Dense, Dropout, Input, Flatten, Lambda, MaxPooling2D, PReLU, 
    ZeroPadding2D
)
from tensorflow.keras.models import Model, Sequential
from numpy import ndarray

from ..commons import functions as F
from ..base.base_models import FacialRecognitionBase, FaceNetBase
from ..commons.settings import models_settings


class ArcFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        """Initialize the ArcFaceClient."""
        self.model = self.load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return self.model(img, training=False).numpy()[0].tolist()

    @classmethod
    def load_model(cls) -> Model:
        """Load the ArcFace model.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_ARCFACE.

        Returns:
            Model: Loaded model."""
        base_model = ArcFaceClient.ResNet34()
        inputs = base_model.inputs[0]
        arcface_model = base_model.outputs[0]
        arcface_model = BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
        arcface_model = Dropout(0.4)(arcface_model)
        arcface_model = Flatten()(arcface_model)
        arcface_model = Dense(512, activation=None, use_bias=True, 
                              kernel_initializer="glorot_normal")(arcface_model)
        embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, 
                                       name="embedding", scale=True)(arcface_model)
        model = Model(inputs, embedding, name=base_model.name)
        model.load_weights(cls._download(models_settings.recognition_attrs.arcface))
        return model

    @staticmethod
    def ResNet34() -> Model:
        """Define the ResNet34 architecture.

        Returns:
            Model: ResNet34 model."""
        img_input = Input(shape=(112, 112, 3))
        x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
        x = Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer="glorot_normal", name="conv1_conv")(x)
        x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
        x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
        x = ArcFaceClient.stack_fn(x)
        return training.Model(img_input, x, name="ResNet34")

    @staticmethod
    def block1(x, filters, kernel_size = 3, stride = 1, conv_shortcut = True, name = None):
        """Define a basic residual block for the ResNet architecture.

        Args:
            x (tensor): Input tensor.
            filters (int): Number of filters.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            stride (int, optional): Stride size. Defaults to 1.
            conv_shortcut (bool, optional): Whether to apply convolutional shortcut. Defaults to True.
            name (str, optional): Block name. Defaults to None.

        Returns:
            tensor: Output tensor."""
        bn_axis = 3
        if conv_shortcut:
            shortcut = Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer="glorot_normal", name=name + "_0_conv")(x)
            shortcut = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_0_bn")(shortcut)
        else:
            shortcut = x
        
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
        x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
        x = Conv2D(filters, 3, strides=1, kernel_initializer="glorot_normal", use_bias=False, name=name + "_1_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
        x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)
        x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
        x = Conv2D(filters, kernel_size, strides=stride, kernel_initializer="glorot_normal", use_bias=False, name=name + "_2_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)
        return Add(name=name + "_add")([shortcut, x])

    @staticmethod
    def stack1(x, filters, blocks, stride1 = 2, name = None):
        """Define a stack of basic residual blocks for the ResNet architecture.

        Args:
            x (tensor): Input tensor.
            filters (int): Number of filters.
            blocks (int): Number of blocks.
            stride1 (int, optional): Stride size for the first block. Defaults to 2.
            name (str, optional): Stack name. Defaults to None.

        Returns:
            tensor: Output tensor."""
        x = ArcFaceClient.block1(x, filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            x = ArcFaceClient.block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
        return x

    @staticmethod
    def stack_fn(x):
        """Define the ResNet architecture.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor."""
        x = ArcFaceClient.stack1(x, 64, 3, name="conv2")
        x = ArcFaceClient.stack1(x, 128, 4, name="conv3")
        x = ArcFaceClient.stack1(x, 256, 6, name="conv4")
        return ArcFaceClient.stack1(x, 512, 3, name="conv5")


class DeepIdClient(FacialRecognitionBase):
    """Initialize the DeepIdClient."""
    def __init__(self) -> None:
        self.model = self.load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return self.model(img, training=False).numpy()[0].tolist()

    @classmethod
    def load_model(cls) -> Model:
        """Load the DeepId model.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_DEEPID.

        Returns:
            Model: Loaded model."""
        myInput = Input(shape=(55, 47, 3))
        x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(myInput)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
        x = Dropout(rate=0.99, name="D1")(x)
        x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
        x = Dropout(rate=0.99, name="D2")(x)
        x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
        x = Dropout(rate=0.99, name="D3")(x)
        x1 = Flatten()(x)
        fc11 = Dense(160, name="fc11")(x1)
        x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name="fc12")(x2)
        y = Add()([fc11, fc12])
        y = Activation("relu", name="deepid")(y)
        model = Model(inputs=[myInput], outputs=y)
        model.load_weights(cls._download(models_settings.recognition_attrs.deepid))
        return model


class FaceNet128dClient(FaceNetBase):
    def __init__(self) -> None:
        """Initialize the FaceNet128dClient."""
        self.model = super().load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return self.model(img, training=False).numpy()[0].tolist()


class FaceNet512dClient(FaceNetBase):
    def __init__(self) -> None:
        """Initialize the FaceNet512dClient."""
        self.model = super().load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return self.model(img, training=False).numpy()[0].tolist()


class OpenFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        """Initialize the OpenFaceClient."""
        self.model = self.load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return self.model(img, training=False).numpy()[0].tolist()

    @classmethod
    def load_model(cls) -> Model:
        """Load the OpenFace model.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_OPENFACE.

        Returns:
            Model: Loaded model."""
        myInput = Input(shape=(96, 96, 3))
        
        x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn1")(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_1")(x)
        x = Conv2D(64, (1, 1), name="conv2")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn2")(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(192, (3, 3), name="conv3")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn3")(x)
        x = Activation("relu")(x)
        x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        
        inception_3a_3x3 = Conv2D(96, (1, 1), name="inception_3a_3x3_conv1")(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_3x3_bn1")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name="inception_3a_3x3_conv2")(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_3x3_bn2")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        
        inception_3a_5x5 = Conv2D(16, (1, 1), name="inception_3a_5x5_conv1")(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_5x5_bn1")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name="inception_3a_5x5_conv2")(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_5x5_bn2")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        
        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name="inception_3a_pool_conv")(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_pool_bn")(inception_3a_pool)
        inception_3a_pool = Activation("relu")(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)
       
        inception_3a_1x1 = Conv2D(64, (1, 1), name="inception_3a_1x1_conv")(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3a_1x1_bn")(inception_3a_1x1)
        inception_3a_1x1 = Activation("relu")(inception_3a_1x1)
        
        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)
        
        inception_3b_3x3 = Conv2D(96, (1, 1), name="inception_3b_3x3_conv1")(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_3x3_bn1")(inception_3b_3x3)
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name="inception_3b_3x3_conv2")(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_3x3_bn2")(inception_3b_3x3)
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
       
        inception_3b_5x5 = Conv2D(32, (1, 1), name="inception_3b_5x5_conv1")(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_5x5_bn1")(inception_3b_5x5)
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name="inception_3b_5x5_conv2")(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_5x5_bn2")(inception_3b_5x5)
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
        
        inception_3b_pool = Lambda(lambda x: x**2, name="power2_3b")(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name="mult9_3b")(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: sqrt(x), name="sqrt_3b")(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name="inception_3b_pool_conv")(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_pool_bn")(inception_3b_pool)
        inception_3b_pool = Activation("relu")(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)
        
        inception_3b_1x1 = Conv2D(64, (1, 1), name="inception_3b_1x1_conv")(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_1x1_bn")(inception_3b_1x1)
        inception_3b_1x1 = Activation("relu")(inception_3b_1x1)
        
        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)
        
        inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name="inception_3c_3x3_conv1")(inception_3b)
        inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_3x3_bn1")(inception_3c_3x3)
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
        inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
        inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name="inception_3c_3x3_conv" + "2")(inception_3c_3x3)
        inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_3x3_bn" + "2")(inception_3c_3x3)
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
        
        inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name="inception_3c_5x5_conv1")(inception_3b)
        inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_5x5_bn1")(inception_3c_5x5)
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
        inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
        inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name="inception_3c_5x5_conv" + "2")(inception_3c_5x5)
        inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_5x5_bn" + "2")(inception_3c_5x5)
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
        
        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)
        
        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)
        
        inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_4a_3x3_conv" + "1")(inception_3c)
        inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "1")(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
        inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
        inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name="inception_4a_3x3_conv" + "2")(inception_4a_3x3)
        inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "2")(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
        
        inception_4a_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name="inception_4a_5x5_conv1")(inception_3c)
        inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_5x5_bn1")(inception_4a_5x5)
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
        inception_4a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
        inception_4a_5x5 = Conv2D(64, (5, 5), strides=(1, 1), name="inception_4a_5x5_conv" + "2")(inception_4a_5x5)
        inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_5x5_bn" + "2")(inception_4a_5x5)
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
        
        inception_4a_pool = Lambda(lambda x: x**2, name="power2_4a")(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name="mult9_4a")(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: sqrt(x), name="sqrt_4a")(inception_4a_pool)
        inception_4a_pool = Conv2D(128, (1, 1), strides=(1, 1), name="inception_4a_pool_conv" + "")(inception_4a_pool)
        inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_pool_bn" + "")(inception_4a_pool)
        inception_4a_pool = Activation("relu")(inception_4a_pool)
        inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)
        
        inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_4a_1x1_conv" + "")(inception_3c)
        inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_1x1_bn" + "")(inception_4a_1x1)
        inception_4a_1x1 = Activation("relu")(inception_4a_1x1)
        
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)
       
        inception_4e_3x3 = Conv2D(160, (1, 1), strides=(1, 1), name="inception_4e_3x3_conv" + "1")(inception_4a)
        inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "1")(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
        inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
        inception_4e_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name="inception_4e_3x3_conv" + "2")(inception_4e_3x3)
        inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "2")(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
       
        inception_4e_5x5 = Conv2D(64, (1, 1), strides=(1, 1), name="inception_4e_5x5_conv" + "1")(inception_4a)
        inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "1")(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
        inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
        inception_4e_5x5 = Conv2D(128, (5, 5), strides=(2, 2), name="inception_4e_5x5_conv" + "2")(inception_4e_5x5)
        inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "2")(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
        
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)
        
        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)
        
        inception_5a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5a_3x3_conv" + "1")(inception_4e)
        inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "1")(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
        inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
        inception_5a_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name="inception_5a_3x3_conv" + "2")(inception_5a_3x3)
        inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "2")(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
       
        inception_5a_pool = Lambda(lambda x: x**2, name="power2_5a")(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name="mult9_5a")(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: sqrt(x), name="sqrt_5a")(inception_5a_pool)
        inception_5a_pool = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5a_pool_conv" + "")(inception_5a_pool)
        inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5a_pool_bn" + "")(inception_5a_pool)
        inception_5a_pool = Activation("relu")(inception_5a_pool)
        inception_5a_pool = ZeroPadding2D(padding=(1, 1))(inception_5a_pool)
        
        inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_5a_1x1_conv" + "")(inception_4e)
        inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5a_1x1_bn" + "")(inception_5a_1x1)
        inception_5a_1x1 = Activation("relu")(inception_5a_1x1)
        
        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)
        
        inception_5b_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5b_3x3_conv" + "1")(inception_5a)
        inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "1")(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
        inception_5b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
        inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name="inception_5b_3x3_conv" + "2")(inception_5b_3x3)
        inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "2")(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
        
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5b_pool_conv" + "")(inception_5b_pool)
        inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5b_pool_bn" + "")(inception_5b_pool)
        inception_5b_pool = Activation("relu")(inception_5b_pool)
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)
       
        inception_5b_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_5b_1x1_conv" + "")(inception_5a)
        inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5b_1x1_bn" + "")(inception_5b_1x1)
        inception_5b_1x1 = Activation("relu")(inception_5b_1x1)
        
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)
        
        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(128, name="dense_layer")(reshape_layer)
        norm_layer = Lambda(lambda x: l2_normalize(x, axis=1), name="norm_layer")(dense_layer)
        model = Model(inputs=[myInput], outputs=norm_layer)
        model.load_weights(cls._download(models_settings.recognition_attrs.openface))
        return model


class VggFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        """Initialize the VggFaceClient."""
        self.model = self.load_model()

    def find_embeddings(self, img: ndarray) -> list[float]:
        """Find facial embeddings from the input image.

        Args:
            img (ndarray): Input image.

        Returns:
            List[float]: Facial embeddings."""
        return F.l2_normalize(self.model(img, training=False).numpy()[0].tolist()).tolist()

    @staticmethod
    def base_model() -> Sequential:
        """Define the base VGGFace model architecture.

        Returns:
            Sequential: VGGFace base model."""
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))
        return model
    
    @classmethod
    def load_model(cls) -> Model:
        """Load the VGGFace model.

        Args:
            url (str, optional): URL for model weights. Defaults to C.DOWNLOAD_URL_VGGFACE.

        Returns:
            Model: Loaded VGGFace model."""
        model = VggFaceClient.base_model()
        model.load_weights(cls._download(models_settings.recognition_attrs.vggface))
        base_model_output = Sequential()
        base_model_output = Flatten()(model.layers[-5].output)
        return Model(inputs=model.input, outputs=base_model_output)
