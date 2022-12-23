import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Average
from tqdm import tqdm
from models import Models

gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.set_visible_devices(gpus[2], 'GPU')
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

path = '/home/turib/test/'


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


model1 = Models().RESNET50()
model2 = Models().RESNET50()
model3 = Models().RESNET50()
model4 = Models().RESNET50()

model1.load_weights(f'{path}weights/0.12-2.63.hdf5')
model2.load_weights(f'{path}weights/90.06-2.76.hdf5')
model3.load_weights(f'{path}weights/180.06-2.71.hdf5')
model4.load_weights(f'{path}weights/270.08-2.69.hdf5')

models = [model1, model2, model3, model4]
model_input = keras.Input(shape=(256, 256, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output, name='ensemble')


def get_data():
    x_test, y_test = [], []
    test_ds = keras.preprocessing.image_dataset_from_directory(
        f'{path}test_data',
        label_mode='categorical',
        image_size=(256, 256),
        batch_size=64)
    for images, labels in tqdm(test_ds.unbatch()):
        x_test.append(images.numpy())
        y_test.append(labels.numpy())
    return np.array(x_test), np.array(y_test)


x_test, y_test = get_data()
x_test = x_test.reshape(-1, 256, 256, 3)


def accuracy(model):
    pred = [model.predict(x_test[i: i + 100], batch_size=8) for i in tqdm(range(0, len(x_test) - 100, 100))]

    # try1 = np.sum(np.not_equal(pred, actual_label)) / y_test.shape[0]
    # try2 = np.sum(np.equal(pred, actual_label)) / y_test.shape[0]
    top3_acc = 0
    top5_acc = 0
    try:
        top3_acc = sum(np.argmax(y_test[i]) in np.argpartition(pred[i], -3)[-3:] for i in range(y_test.shape[0])) / \
                   y_test.shape[0]
        top5_acc = sum(np.argmax(y_test[i]) in np.argpartition(pred[i], -5)[-5:] for i in range(y_test.shape[0])) / \
                   y_test.shape[0]
    except Exception:
        pass
    return sum(np.argmax(pred[i]) == np.argmax(y_test[i]) for i in range(y_test.shape[0])) / y_test.shape[
        0], top3_acc, top5_acc


print(f"Ensemble model acc: {accuracy(ensemble_model)}")