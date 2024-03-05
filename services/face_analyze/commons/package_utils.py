from tensorflow.keras.backend import int_shape, sqrt, l2_normalize
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Add, add, BatchNormalization, Concatenate, 
    concatenate, Conv2D, Convolution2D, Dense, Dropout, Input, Flatten, 
    GlobalAveragePooling2D, Lambda, LocallyConnected2D, MaxPooling2D, PReLU, 
    ZeroPadding2D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image