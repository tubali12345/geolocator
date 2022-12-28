from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from pathlib import Path
from datetime import date
from data import Data
import tensorflow as tf
from models import RESNET50, RESNET101

path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'


def callbacks(filepath: str, model_type: str) -> list:
    logdir = f'{path}logs/{model_type}_{date.today()}'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=1, min_lr=0)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_SGD(model, lr=1e-4, momentum=0.9):
    return model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])


def train(model, train_data: tf.data.Dataset, val_data: tf.data.Dataset, num_epochs: int, model_name: str, heading=None):
    outdir = Path(f'{path}weights/{model_name}_{date.today()}')
    outdir.mkdir(exist_ok=True, parents=True)
    if heading is not None:
        filepath = f'{path}weights/{model_name}_{date.today()}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'
    else:
        filepath = f'{path}weights/{model_name}_{date.today()}/' + '.{epoch:02d}-{loss:.2f}.hdf5'
    return model.fit(train_data,  validation_data=val_data, epochs=num_epochs, callbacks=callbacks(filepath, model_name))


def train_ensemble(create_model_func, model_name: str, num_epochs=50) -> dict:
    models = {heading: create_model_func() for heading in [0, 90, 180, 270]}
    histories = {}

    for heading, model in models.items():
        compile_SGD(model)
        data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123)
        train_data = data.load_train(f'pictures_{heading}')
        val_data = data.load_val(f'pictures_{heading}')
        histories[heading] = train(model, train_data, val_data, num_epochs, heading, model_name)
    return histories


def train_resnet50(model_name):
    model = RESNET50()
    compile_SGD(model)
    data = Data(validation_split=0.15, image_size=(256, 256), batch_size=8, seed=123)
    train_data = data.load_train('data')
    val_data = data.load_val('data')
    return train(model, train_data, val_data, num_epochs=50, model_name=model_name)


if __name__ == '__main__':
    train_resnet50('1RESNET50')
    # train_ensemble(RESNET50, 'resnet50 ensemble')
    # train_ensemble(RESNET101, 'resnet101')
