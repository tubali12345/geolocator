import contextlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Average
from tqdm import tqdm
from models import RESNET50

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

path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/weights/only_resnet50/'


model1 = RESNET50()
model2 = RESNET50()
model3 = RESNET50()
model4 = RESNET50()

model1.load_weights(f'{path}0.06-1.48.hdf5')
model2.load_weights(f'{path}90.06-1.54.hdf5')
model3.load_weights(f'{path}180.06-1.54.hdf5')
model4.load_weights(f'{path}270.04-2.02.hdf5')

models = [model1, model2, model3, model4]
model_input = keras.Input(shape=(256, 256, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output, name='ensemble')


def get_data():
    p = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'
    x_test, y_test = [], []
    test_ds = keras.preprocessing.image_dataset_from_directory(
        f'{p}test_data',
        label_mode='categorical',
        image_size=(256, 256),
        validation_split=0.002,
        seed=123,
        shuffle=True,
        subset='validation',
        batch_size=8)
    for images, labels in tqdm(test_ds.unbatch()):
        x_test.append(images.numpy())
        y_test.append(labels.numpy())
    return np.array(x_test), np.array(y_test)


x_test, y_test = get_data()
x_test = x_test.reshape(-1, 256, 256, 3)


def topk_accuracy(model, list_k: list, patch_size=100):
    pred = []
    for i in tqdm(range(0, len(x_test) - patch_size, patch_size)):
        pred.extend(model.predict(x_test[i: i + patch_size], batch_size=8))
    pred.extend(model.predict(x_test[len(x_test) - patch_size:], batch_size=8))
    pred = np.array(pred)
    return {k: sum(np.argmax(y_test[i]) in np.argpartition(pred[i], -k)[-k:] for i in range(pred.shape[0])) / pred.shape[0] for k in list_k}


accuracies = topk_accuracy(ensemble_model, [1, 3, 5])


def print_accuracy(accuracies, model_name):
    print(f"{model_name} model accuracy:")
    for k, acc in accuracies.items():
        print(f'Top {k}: {acc}')

print_accuracy(accuracies, 'RESNET50 ensemble')
