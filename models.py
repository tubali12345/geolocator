from tensorflow import keras
from keras.layers import Dense, Flatten, Average


def RESNET50():
    model = keras.Sequential()
    resnet50_model = keras.applications.ResNet50(include_top=False,
                                                 input_shape=(256, 256, 3),
                                                 pooling='avg', classes=50)
    model.add(resnet50_model)
    model.add(Flatten())
    model.add(Dense(50, activation='softmax'))
    model.summary()
    return model


def RESNET101():
    model = keras.Sequential()

    resnet101_model = keras.applications.ResNet101(include_top=False, input_shape=(256, 256, 3),
                                                   pooling='avg', classes=50)
    model.add(resnet101_model)
    model.add(Flatten())
    model.add(Dense(50, activation='softmax'))
    model.summary()
    return model


def ensemble_model(model, weight_paths: dict):
    def_models = {heading: model() for heading in [0, 90, 180, 270]}
    for heading in def_models:
        def_models[heading].load_weights(weight_paths[heading])
    models = list(def_models.values())
    model_input = keras.Input(shape=(256, 256, 3))
    model_outputs = [model(model_input) for model in models]
    ensemble_output = Average()(model_outputs)
    return keras.Model(inputs=model_input, outputs=ensemble_output, name='ensemble')
