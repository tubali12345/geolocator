from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Average
import numpy as np
from keras.optimizers import SGD
from pathlib import Path
from datetime import date
from data import Data
import tensorflow as tf

path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'


def create_model():
    resnet_model = keras.Sequential()
    pretrained_model = keras.applications.ResNet50(include_top=False,
                                                   input_shape=(256, 256, 3),
                                                   pooling='avg', classes=50,
                                                   weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(50, activation='softmax'))
    return resnet_model


def create_model_o_resnet():
    resnet_model = keras.Sequential()

    pretrained_model = keras.applications.ResNet50(include_top=False,
                                                   input_shape=(256, 256, 3),
                                                   pooling='avg', classes=50)

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(50, activation='softmax'))
    return resnet_model


def create_model_o_resnet101():
    resnet_model = keras.Sequential()

    pretrained_model = keras.applications.ResNet101(include_top=False, input_shape=(256, 256, 3),
                                                    pooling='avg', classes=50)

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(50, activation='softmax'))
    return resnet_model


def create_model_resnet101():
    resnet_model = keras.Sequential()

    pretrained_model = keras.applications.ResNet50(include_top=False,
                                                   input_shape=(256, 256, 3),
                                                   pooling='avg', classes=50,
                                                   weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(50, activation='softmax'))
    return resnet_model


def callbacks(filepath: str, model_type: str) -> list:
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=f'logs/{model_type}_{date.today()}')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.0005)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_and_train_SGD(model, num_epochs: int, heading: int, model_type: str, lr=1e-4):
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    outdir = Path(f'{path}weights/{model_type}')
    outdir.mkdir(exist_ok=True, parents=True)
    filepath = f'{path}weights/{model_type}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'

    data = Data(validation_split=0.2, image_size=(256, 256), batch_size=8, seed=123)
    return model.fit(data.load_train(f'pictures_{heading}'), validation_data=data.load_test(f'pictures_{heading}'),
                     epochs=num_epochs, callbacks=callbacks(filepath, model_type))


def main(create_model_func, compile_and_train_func, model_type):
    # model1 = create_model_func()
    # model2 = create_model_func()
    # model3 = create_model_func()
    model4 = create_model_func()

    NUM_EPOCHS = 50

    # history1 = compile_and_train_func(model1, NUM_EPOCHS, 0, model_type)
    # history2 = compile_and_train_func(model2, NUM_EPOCHS, 90, model_type)
    # history3 = compile_and_train_func(model3, NUM_EPOCHS, 180, model_type)
    history4 = compile_and_train_func(model4, NUM_EPOCHS, 270, model_type)

    # return history1, history2, history3, history4


if __name__ == '__main__':
    # main(create_model, compile_and_train_SGD, 'resnet50')
    main(create_model_o_resnet, compile_and_train_SGD, 'only_resnet50')
    # main(create_model_resnet101, compile_and_train_SGD, 'resnet101')
    # main(create_model_o_resnet101, compile_and_train_SGD, 'only_resnet101')

    # main(create_model_pt_resnet, compile_and_train_SGD, 'pt_resnet50')
