from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

from data import Data
from models import RESNET50, ViT, ensemble_model_perceptron
from utils import Config, Parameters


def callbacks(model_name: str,
              heading: int = None) -> list:
    weights_dir = Path(f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}')
    weights_dir.mkdir(exist_ok=True, parents=True)
    if heading is not None:
        filepath = f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'
        logdir = f'{Config.PATH}logs/{model_name}_{heading}_{date.today()}'
    else:
        filepath = f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}/' + '.{epoch:02d}-{loss:.2f}.hdf5'
        logdir = f'{Config.PATH}logs/{model_name}_{date.today()}'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=Parameters.verbose, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=Parameters.early_stop_patience, verbose=Parameters.verbose)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=Parameters.reduce_lr_factor,
                                  patience=Parameters.reduce_lr_patience, min_lr=Parameters.min_lr)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_SGD(model,
                lr: float = Parameters.learning_rate,
                momentum: float = Parameters.momentum):
    return model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])


def train(model,
          train_data: tf.data.Dataset,
          val_data: tf.data.Dataset,
          num_epochs: int,
          model_name: str,
          heading=None):
    return model.fit(train_data,  validation_data=val_data, epochs=num_epochs, callbacks=callbacks(model_name, heading))


def train_ensemble(create_model_func,
                   model_name: str,
                   num_epochs: int = Parameters.num_epochs) -> dict:
    data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    models = {heading: create_model_func(input_shape) for heading in [0, 90, 180, 270]}
    histories = {}

    for heading, model in models.items():
        compile_SGD(model)
        train_data = data.load_train(f'pictures_{heading}')
        val_data = data.load_val(f'pictures_{heading}')
        histories[heading] = train(model, train_data, val_data, num_epochs, model_name, heading)
    return histories


def train_single(create_model_func,
                 model_name: str,
                 num_epochs: int = Parameters.num_epochs):
    data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    model = create_model_func(input_shape)
    compile_SGD(model)
    train_data = data.load_train('data')
    val_data = data.load_val('data')
    return train(model, train_data, val_data, num_epochs=num_epochs, model_name=model_name)


def train_ViT(model_name: str,
              num_epochs: int = Parameters.num_epochs):
    data = Data(validation_split=0, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    ds = data.load_train('data')
    x_train, y_train = [], []
    for images, labels in ds.unbatch():
        x_train.append(images.numpy())
        y_train.append(labels.numpy())
    x_train, y_train = np.array(x_train), np.array(y_train)
    model = ViT(input_shape, x_train)
    compile_SGD(model)
    return model.fit(x=x_train, y=y_train, epochs=num_epochs, validation_split=0.15, callbacks=callbacks(model_name))


def train_ensemble_perceptron(model_name: str,
                              num_epochs: int = Parameters.num_epochs):
    data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)

    w_dir = Config.WEIGHTS_PATH
    w_paths = {0: f'{w_dir}resnet50 ensemble_2022-12-29/0.09-1.16.hdf5',
               90: f'{w_dir}resnet50 ensemble_2022-12-30/90.09-1.18.hdf5',
               180: f'{w_dir}resnet50 ensemble_2022-12-30/180.09-1.17.hdf5',
               270: f'{w_dir}resnet50 ensemble_2022-12-31/270.09-1.22.hdf5'}
    e_model = ensemble_model_perceptron(RESNET50, input_shape, w_paths)
    compile_SGD(e_model)
    return train(e_model, data.load_train('e_data'), data.load_val('e_data'), num_epochs=num_epochs, model_name=model_name)


if __name__ == '__main__':
    train_ensemble_perceptron('new_ensemble')
    # train_ViT('ViT')
    # train_single(RESNET50, '1RESNET50')
    # train_ensemble(RESNET50, 'resnet50 ensemble')

