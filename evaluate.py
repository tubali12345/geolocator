import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import RESNET50, ensemble_model
from data import Data
from pathlib import Path
from utils import Config
from datetime import date


def predict(model, x_test, patch_size=100) -> np.array:
    pred = []
    for i in tqdm(range(0, len(x_test) - patch_size, patch_size)):
        pred.extend(model.predict(x_test[i: i + patch_size], batch_size=8))
    pred.extend(model.predict(x_test[len(x_test) - patch_size:], batch_size=8))
    return np.array(pred)


def topk_accuracy(pred: np.array, test_label: np.array, list_k: list[int]) -> dict[int: float]:
    return {
        k: sum(np.argmax(test_label[i]) in np.argpartition(pred[i], -k)[-k:] for i in range(pred.shape[0])) /
           pred.shape[0] for k in list_k}


def accuracy_by_state(pred: np.array, test_label: np.array, k=1) -> dict[int: float]:
    assert pred.shape[0] == test_label.shape[0]
    return {
        state: sum(np.argmax(test_label[i]) in np.argpartition(pred[i], -k)[-k:] if np.argmax(test_label[i]) == state else 0 for i in
               range(pred.shape[0])) / sum(np.argmax(test_label[i]) == state for i in range(test_label.shape[0])) for state in range(50)}


def print_accuracy(accuracies: dict[int: float], model_name: str, acc_by_state=False, write_to_file=False) -> None:
    print(f"{model_name} model accuracy:")
    for k, acc in accuracies.items():
        if acc_by_state:
            print(f'{Data.states[k]}: {acc}')
            if write_to_file:
                Path(f'{Config.PATH}{model_name}_{date.today()}_accuracy.txt').open('a').write(f'{Data.states[k]}: {acc}\n')
        else:
            print(f'Top {k}: {acc}')


if __name__ == '__main__':
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
    data = Data(validation_split=0.07, image_size=(256, 256), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    x_test, y_test = data.load_test()

    ''' RESNET 50 model'''

    # w_dir0 = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/weights/1RESNET50_2022-12-28/'
    # resnet50_model = RESNET50(input_shape)
    # resnet50_model.load_weights(f'{w_dir0}.07-1.03.hdf5')
    #
    # predictions_r50 = predict(resnet50_model, x_test)
    # print_accuracy(accuracy_by_state(predictions_r50, y_test), 'RESNET50 model', acc_by_state=True, write_to_file=True)
    # print_accuracy(topk_accuracy(predictions_r50, y_test, [1, 3, 5]), 'RESNET50 model')

    ''' RESNET 50 ENSEMBLE model'''

    w_dir = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/weights/'
    w_paths = {0: f'{w_dir}resnet50 ensemble_2022-12-29/0.09-1.16.hdf5',
               90: f'{w_dir}resnet50 ensemble_2022-12-30/90.09-1.18.hdf5',
               180: f'{w_dir}resnet50 ensemble_2022-12-30/180.09-1.17.hdf5',
               270: f'{w_dir}resnet50 ensemble_2022-12-31/270.09-1.22.hdf5'}
    ensemble_model = ensemble_model(RESNET50, input_shape, w_paths)

    predictions_r50e = predict(ensemble_model, x_test)
    print_accuracy(accuracy_by_state(predictions_r50e, y_test), 'RESNET50 ensemble model', acc_by_state=True, write_to_file=True)
    print_accuracy(topk_accuracy(predictions_r50e, y_test, [1, 3, 5]), 'RESNET50 ensemble model')
