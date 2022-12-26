from tensorflow import keras
from keras.layers import Dense, Flatten


def RESNET50():
    resnet_model = keras.Sequential()

    pretrained_model = keras.applications.ResNet50(include_top=False,
                                                   input_shape=(256, 256, 3),
                                                   pooling='avg', classes=50)

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(50, activation='softmax'))
    resnet_model.summary()
    return resnet_model


def RESNET101():
    resnet_model = keras.Sequential()

    pretrained_model = keras.applications.ResNet101(include_top=False, input_shape=(256, 256, 3),
                                                    pooling='avg', classes=50)

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(50, activation='softmax'))
    return resnet_model
