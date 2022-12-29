from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from pathlib import Path
from datetime import date
from data import Data
import tensorflow as tf
from models import RESNET50, RESNET101, ViT
from utils import Config, TrainParameters
import numpy as np


def callbacks(model_name: str, heading: int = None) -> list:
    weights_dir = Path(f'{Config.PATH}weights/{model_name}_{date.today()}')
    weights_dir.mkdir(exist_ok=True, parents=True)
    if heading is not None:
        filepath = f'{Config.PATH}weights/{model_name}_{date.today()}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'
    else:
        filepath = f'{Config.PATH}weights/{model_name}_{date.today()}/' + '.{epoch:02d}-{loss:.2f}.hdf5'
    logdir = f'{Config.PATH}logs/{model_name}_{date.today()}'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=TrainParameters.verbose, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=TrainParameters.early_stop_patience, verbose=TrainParameters.verbose)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=TrainParameters.reduce_lr_factor,
                                  patience=TrainParameters.reduce_lr_patience, min_lr=TrainParameters.min_lr)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_SGD(model, lr: float = TrainParameters.learning_rate, momentum: float = TrainParameters.momentum):
    return model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])


def train(model, train_data: tf.data.Dataset, val_data: tf.data.Dataset, num_epochs: int, model_name: str, heading=None):
    return model.fit(train_data,  validation_data=val_data, epochs=num_epochs, callbacks=callbacks(model_name, heading))


def train_ensemble(create_model_func, model_name: str, num_epochs: int = TrainParameters.num_epochs) -> dict:
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


def train_single(create_model_func, model_name: str, num_epochs: int = TrainParameters.num_epochs):
    data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    model = create_model_func(input_shape)
    compile_SGD(model)
    train_data = data.load_train('data')
    val_data = data.load_val('data')
    return train(model, train_data, val_data, num_epochs=num_epochs, model_name=model_name)


def train_ViT(model_name: str, num_epochs: int = TrainParameters.num_epochs):
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
    return model.fit(x=x_train, y=y_train, epochs=num_epochs, validation_split=0.1, callbacks=callbacks(model_name))


if __name__ == '__main__':
    # train_ViT('ViT')
    # train_single(RESNET50, '1RESNET50')
    train_ensemble(RESNET50, 'resnet50 ensemble')
    # train_ensemble(RESNET101, 'resnet101')
