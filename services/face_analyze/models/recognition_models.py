from typing import List
from os.path import isfile

import tensorflow as tf
from cv2 import FaceRecognizerSF
from zipfile import ZipFile
from bz2 import BZ2File
from dlib import face_recognition_model_v1
from gdown import download
from numpy import array, expand_dims, ndarray, uint8

import commons.constants as C
import commons.functions as F
from base.base_models import FacialRecognitionBase, FaceNetBase
from commons.folder_utils import get_deepface_home
from commons.package_utils import (
    Model, BatchNormalization, Dropout, Flatten, Dense, Input, ZeroPadding2D, Conv2D, 
    PReLU, Add, Sequential, Convolution2D, MaxPooling2D, Activation, training, 
    LocallyConnected2D, Lambda, concatenate, AveragePooling2D, l2_normalize, sqrt
)

class ArcFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "ArcFace"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()

    def load_model(self, url: str = C.DOWNLOAD_URL_ARCFACE) -> Model:
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
        output = get_deepface_home() + C.PATH_WEIGHTS_ARCFACE
        FacialRecognitionBase._download(url, output)
        model.load_weights(output)
        return model

    @staticmethod
    def ResNet34() -> Model:
        img_input = Input(shape=(112, 112, 3))
        x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
        x = Conv2D(64, 3, strides=1, use_bias=False, 
                   kernel_initializer="glorot_normal", name="conv1_conv")(x)
        x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
        x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
        x = ArcFaceClient.stack_fn(x)
        return training.Model(img_input, x, name="ResNet34")

    @staticmethod
    def block1(x, filters, kernel_size = 3, stride = 1, conv_shortcut = True, name = None):
        bn_axis = 3
        if conv_shortcut:
            shortcut = Conv2D(filters,
                              1,
                              strides=stride,
                              use_bias=False,
                              kernel_initializer="glorot_normal",
                              name=name + "_0_conv")(x)
            shortcut = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, 
                                          name=name + "_0_bn")(shortcut)
        else:
            shortcut = x
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
        x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
        x = Conv2D(filters,
                   3,
                   strides=1,
                   kernel_initializer="glorot_normal",
                   use_bias=False,
                   name=name + "_1_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
        x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)
        x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
        x = Conv2D(filters,
                   kernel_size,
                   strides=stride,
                   kernel_initializer="glorot_normal",
                   use_bias=False,
                   name=name + "_2_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)
        return Add(name=name + "_add")([shortcut, x])

    @staticmethod
    def stack1(x, filters, blocks, stride1 = 2, name = None):
        x = ArcFaceClient.block1(x, filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            x = ArcFaceClient.block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
        return x

    @staticmethod
    def stack_fn(x):
        x = ArcFaceClient.stack1(x, 64, 3, name="conv2")
        x = ArcFaceClient.stack1(x, 128, 4, name="conv3")
        x = ArcFaceClient.stack1(x, 256, 6, name="conv4")
        return ArcFaceClient.stack1(x, 512, 3, name="conv5")


class DeepFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "DeepFace"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()

    def load_model(self, url: str = C.DOWNLOAD_URL_DEEPFACE) -> Model:
        base_model = Sequential()
        base_model.add(Convolution2D(32, (11, 11), activation="relu", 
                                     name="C1", input_shape=(152, 152, 3)))
        base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
        base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
        base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
        base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5"))
        base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
        base_model.add(Flatten(name="F0"))
        base_model.add(Dense(4096, activation="relu", name="F7"))
        base_model.add(Dropout(rate=0.5, name="D0"))
        base_model.add(Dense(8631, activation="softmax", name="F8"))
        home = get_deepface_home()
        dr = home + C.PATH_WEIGHTS_DEEPFACE
        if not isfile(dr):
            output = dr + ".zip"
            download(url, output)
            with ZipFile(output, "r") as zip_ref:
                zip_ref.extractall(home + C._WEIGHTS)
        base_model.load_weights(dr)
        return Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)


class DeepIdClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "DeepId"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()

    def load_model(self, url: str = C.DOWNLOAD_URL_DEEPID) -> Model:
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
        output = get_deepface_home() + C.PATH_WEIGHTS_DEEPID
        self._download(url, output)
        model.load_weights(output)
        return model


class DlibResNet:
    def __init__(self) -> None:
        self.layers = [DlibMetaData()]
        file = get_deepface_home() + C.PATH_WEIGHTS_DLIB
        if not isfile(file):
            output = f"{file}.bz2"
            download(f"http://dlib.net/files/{C.DLIB_NAME}.bz2", output)
            with open(output[:-4], "wb") as f:
                f.write(BZ2File(output).read())
        self.model = face_recognition_model_v1(file)


class DlibMetaData:
    def __init__(self) -> None: 
        self.input_shape = [[1, 150, 150, 3]]


class DlibClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = DlibResNet(), "Dlib"

    def find_embeddings(self, img: ndarray) -> List[float]:
        if len(img.shape) == 4:
            img = img[0]

        img = img[:, :, ::-1]

        if img.max() <= 1:
            img = img * 255

        return expand_dims(array(self.model.model.compute_face_descriptor(
            img.astype(uint8))), axis=0)[0].tolist()


class FaceNet128dClient(FaceNetBase):
    def __init__(self) -> None:
        self.model, self.model_name = super().load_model(), "FaceNet-128d"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()


class FaceNet512dClient(FaceNetBase):
    def __init__(self) -> None:
        self.model, self.model_name = super().load_model(), "FaceNet-512d"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()


class OpenFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "OpenFace"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model(img, training=False).numpy()[0].tolist()

    def load_model(self, url: str = C.DOWNLOAD_URL_OPENFACE) -> Model:
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
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_3x3_bn1")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name="inception_3a_3x3_conv2")(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_3x3_bn2")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        inception_3a_5x5 = Conv2D(16, (1, 1), name="inception_3a_5x5_conv1")(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_5x5_bn1")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name="inception_3a_5x5_conv2")(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_5x5_bn2")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name="inception_3a_pool_conv")(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_3a_pool_bn")(inception_3a_pool)
        inception_3a_pool = Activation("relu")(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)
        inception_3a_1x1 = Conv2D(64, (1, 1), name="inception_3a_1x1_conv")(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_1x1_bn")(inception_3a_1x1)
        inception_3a_1x1 = Activation("relu")(inception_3a_1x1)
        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, 
                                    inception_3a_pool, inception_3a_1x1], axis=3)
        inception_3b_3x3 = Conv2D(96, (1, 1), name="inception_3b_3x3_conv1")(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3b_3x3_bn1")(inception_3b_3x3)
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name="inception_3b_3x3_conv2")(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3b_3x3_bn2")(inception_3b_3x3)
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
        inception_3b_5x5 = Conv2D(32, (1, 1), name="inception_3b_5x5_conv1")(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3b_5x5_bn1")(inception_3b_5x5)
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name="inception_3b_5x5_conv2")(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3b_5x5_bn2")(inception_3b_5x5)
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
        inception_3b_pool = Lambda(lambda x: x**2, name="power2_3b")(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name="mult9_3b")(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: sqrt(x), name="sqrt_3b")(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name="inception_3b_pool_conv")(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_3b_pool_bn")(inception_3b_pool)
        inception_3b_pool = Activation("relu")(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)
        inception_3b_1x1 = Conv2D(64, (1, 1), name="inception_3b_1x1_conv")(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3b_1x1_bn")(inception_3b_1x1)
        inception_3b_1x1 = Activation("relu")(inception_3b_1x1)
        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, 
                                    inception_3b_pool, inception_3b_1x1], axis=3)
        inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), 
                                  name="inception_3c_3x3_conv1")(inception_3b)
        inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3c_3x3_bn1")(inception_3c_3x3)
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
        inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
        inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), 
                                  name="inception_3c_3x3_conv" + "2")(inception_3c_3x3)
        inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3c_3x3_bn" + "2")(inception_3c_3x3)
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
        inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), 
                                  name="inception_3c_5x5_conv1")(inception_3b)
        inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3c_5x5_bn1")(inception_3c_5x5)
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
        inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
        inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), 
                                  name="inception_3c_5x5_conv" + "2")(inception_3c_5x5)
        inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3c_5x5_bn" + "2")(inception_3c_5x5)
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)
        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)
        inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), 
                                  name="inception_4a_3x3_conv" + "1")(inception_3c)
        inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4a_3x3_bn" + "1")(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
        inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
        inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), 
                                  name="inception_4a_3x3_conv" + "2")(inception_4a_3x3)
        inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "2")(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
        inception_4a_5x5 = Conv2D(32, (1, 1), strides=(1, 1), 
                                  name="inception_4a_5x5_conv1")(inception_3c)
        inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4a_5x5_bn1")(inception_4a_5x5)
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
        inception_4a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
        inception_4a_5x5 = Conv2D(64, (5, 5), strides=(1, 1), 
                                  name="inception_4a_5x5_conv" + "2")(inception_4a_5x5)
        inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4a_5x5_bn" + "2")(inception_4a_5x5)
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
        inception_4a_pool = Lambda(lambda x: x**2, name="power2_4a")(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name="mult9_4a")(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: sqrt(x), name="sqrt_4a")(inception_4a_pool)
        inception_4a_pool = Conv2D(128, (1, 1), strides=(1, 1), 
                                   name="inception_4a_pool_conv" + "")(inception_4a_pool)
        inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_4a_pool_bn" + "")(inception_4a_pool)
        inception_4a_pool = Activation("relu")(inception_4a_pool)
        inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)
        inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), 
                                  name="inception_4a_1x1_conv" + "")(inception_3c)
        inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4a_1x1_bn" + "")(inception_4a_1x1)
        inception_4a_1x1 = Activation("relu")(inception_4a_1x1)
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, 
                                    inception_4a_pool, inception_4a_1x1], axis=3)
        inception_4e_3x3 = Conv2D(160, (1, 1), strides=(1, 1), 
                                  name="inception_4e_3x3_conv" + "1")(inception_4a)
        inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4e_3x3_bn" + "1")(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
        inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
        inception_4e_3x3 = Conv2D(256, (3, 3), strides=(2, 2), 
                                  name="inception_4e_3x3_conv" + "2")(inception_4e_3x3)
        inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4e_3x3_bn" + "2")(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
        inception_4e_5x5 = Conv2D(64, (1, 1), strides=(1, 1), 
                                  name="inception_4e_5x5_conv" + "1")(inception_4a)
        inception_4e_5x5 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "1"
        )(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
        inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
        inception_4e_5x5 = Conv2D(128, (5, 5), strides=(2, 2), 
                                  name="inception_4e_5x5_conv" + "2")(inception_4e_5x5)
        inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_4e_5x5_bn" + "2")(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)
        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)
        inception_5a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), 
                                  name="inception_5a_3x3_conv" + "1")(inception_4e)
        inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5a_3x3_bn" + "1")(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
        inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
        inception_5a_3x3 = Conv2D(384, (3, 3), strides=(1, 1), 
                                  name="inception_5a_3x3_conv" + "2")(inception_5a_3x3)
        inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5a_3x3_bn" + "2")(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
        inception_5a_pool = Lambda(lambda x: x**2, name="power2_5a")(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name="mult9_5a")(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: sqrt(x), name="sqrt_5a")(inception_5a_pool)
        inception_5a_pool = Conv2D(96, (1, 1), strides=(1, 1), 
                                   name="inception_5a_pool_conv" + "")(inception_5a_pool)
        inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_5a_pool_bn" + "")(inception_5a_pool)
        inception_5a_pool = Activation("relu")(inception_5a_pool)
        inception_5a_pool = ZeroPadding2D(padding=(1, 1))(inception_5a_pool)
        inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), 
                                  name="inception_5a_1x1_conv" + "")(inception_4e)
        inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5a_1x1_bn" + "")(inception_5a_1x1)
        inception_5a_1x1 = Activation("relu")(inception_5a_1x1)
        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)
        inception_5b_3x3 = Conv2D(96, (1, 1), strides=(1, 1), 
                                  name="inception_5b_3x3_conv" + "1")(inception_5a)
        inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5b_3x3_bn" + "1")(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
        inception_5b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
        inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), 
                                  name="inception_5b_3x3_conv" + "2")(inception_5b_3x3)
        inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5b_3x3_bn" + "2")(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = Conv2D(96, (1, 1), strides=(1, 1), 
                                   name="inception_5b_pool_conv" + "")(inception_5b_pool)
        inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_5b_pool_bn" + "")(inception_5b_pool)
        inception_5b_pool = Activation("relu")(inception_5b_pool)
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)
        inception_5b_1x1 = Conv2D(256, (1, 1), strides=(1, 1), 
                                  name="inception_5b_1x1_conv" + "")(inception_5a)
        inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_5b_1x1_bn" + "")(inception_5b_1x1)
        inception_5b_1x1 = Activation("relu")(inception_5b_1x1)
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, 
                                    inception_5b_1x1], axis=3)
        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(128, name="dense_layer")(reshape_layer)
        norm_layer = Lambda(lambda x: l2_normalize(x, axis=1), name="norm_layer")(dense_layer)
        model = Model(inputs=[myInput], outputs=norm_layer)
        output = get_deepface_home() + C.PATH_WEIGHTS_OPENFACE
        self._download(url, output)
        model.load_weights(output)
        return model


class SFaceWrapper:
    class Layer:
        input_shape, output_shape = (None, 112, 112, 3), (None, 1, 128)

    def __init__(self, model_path: str) -> None:
        self.model, self.layers = FaceRecognizerSF.create(model_path, "", 0, 0), [self.Layer()]


class SFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "SFace"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return self.model.model.feature((img[0] * 255).astype(uint8))[0].tolist()

    def load_model(self, url: str = C.DOWNLOAD_URL_SFACE) -> SFaceWrapper:
        output = get_deepface_home() + C.PATH_WEIGHTS_SFACE
        self._download(url, output)
        return SFaceWrapper(output)


class VggFaceClient(FacialRecognitionBase):
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "VGG-Face"

    def find_embeddings(self, img: ndarray) -> List[float]:
        return F.l2_normalize(self.model(img, training=False).numpy()[0].tolist()).tolist()

    @staticmethod
    def base_model() -> Sequential:
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

    def load_model(self, url: str = C.DOWNLOAD_URL_VGGFACE) -> Model:
        model = VggFaceClient.base_model()
        output = get_deepface_home() + C.PATH_WEIGHTS_VGGFACE
        self._download(url, output)
        model.load_weights(output)
        base_model_output = Sequential()
        base_model_output = Flatten()(model.layers[-5].output)
        return Model(inputs=model.input, outputs=base_model_output)
