from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Activation

def create_layer(model,num_filters):
    model.add(Conv2D(num_filters, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_filters, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    return model

def create_net(num_layer_32,num_layer_64,num_layer_128,num_layer_256,num_layer_512,InputShape):
    model = Sequential()
    model.add(Conv2D(64,(3,3),padding="same",input_shape=InputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    for i in range(num_layer_64):
        create_layer(model,64)
    for i in range(num_layer_128):
        create_layer(model,128)
    for i in range(num_layer_64):
        create_layer(model,64)
    for i in range(num_layer_32):
        create_layer(model,32)
    # for i in range(num_layer_256):
    #     create_layer(model,256)
    # for i in range(num_layer_512):
    #     create_layer(model,512)
# final conv-layer with 1 filter:
    model.add(Conv2D(1, (3, 3), padding="same"))
    print(model.summary())
    return model

