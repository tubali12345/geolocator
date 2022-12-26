from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Average
import numpy as np
from keras.optimizers import SGD
from pathlib import Path
from datetime import date
from data import Data
import tensorflow as tf
from models import RESNET50, RESNET101

path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'


def callbacks(filepath: str, model_type: str) -> list:
    logdir = f'{path}logs/{model_type}_{date.today()}'
    train_writer = tf.summary.create_file_writer(f'{logdir}/batch_level') #batch-level logging, need to add
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.0005)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_SGD(model, lr=1e-4, momentum=0.9):
    return model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])


def train(model, train_data, val_data, num_epochs: int, heading: int, model_name: str):
    outdir = Path(f'{path}weights/{model_name}')
    outdir.mkdir(exist_ok=True, parents=True)
    filepath = f'{path}weights/{model_name}_{date.today()}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'
    return model.fit(train_data,  validation_data=val_data, epochs=num_epochs, callbacks=callbacks(filepath, model_name))


def train_ensemble(create_model_func, model_name: str, num_epochs=50):
    model1 = create_model_func()
    model2 = create_model_func()
    model3 = create_model_func()
    model4 = create_model_func()

    models = {0: model1, 90: model2, 180: model3, 270: model4}
    histories = []

    for heading, model in models.items():
        compile_SGD(model)
        data = Data(validation_split=0.99, image_size=(256, 256), batch_size=8, seed=124)
        train_data = data.load_train(f'pictures_{heading}')
        val_data = data.load_test(f'pictures_{heading}')
        histories.append(train(model, train_data, val_data, num_epochs, heading, model_name))
    return histories


if __name__ == '__main__':
    train_ensemble(RESNET50, 'resnet50')
    # train_ensemble(RESNET101, 'resnet101')