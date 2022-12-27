import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import RESNET50, ensemble_model
from data import Data


def topk_accuracy(model, list_k: list[int], patch_size=100) -> dict[int: float]:
    pred = []
    for i in tqdm(range(0, len(x_test) - patch_size, patch_size)):
        pred.extend(model.predict(x_test[i: i + patch_size], batch_size=8))
    pred.extend(model.predict(x_test[len(x_test) - patch_size:], batch_size=8))
    pred = np.array(pred)
    return {
        k: sum(np.argmax(y_test[i]) in np.argpartition(pred[i], -k)[-k:] for i in range(pred.shape[0])) / pred.shape[0]
        for k in list_k}


def print_accuracy(accuracies: dict[int: float], model_name: str) -> None:
    print(f"{model_name} model accuracy:")
    for k, acc in accuracies.items():
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
    data = Data(validation_split=0.002, image_size=(256, 256), batch_size=8, seed=123)
    x_test, y_test = data.load_test()

    ''' RESNET 50 model'''

    w_dir0 = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/weights/only_resnet50/'
    resnet50_model = RESNET50()
    resnet50_model.load_weights(f'{w_dir0}0.06-1.48.hdf5')
    accuracies0 = topk_accuracy(resnet50_model, [1, 3, 5])
    print_accuracy(accuracies0, 'RESNET50 model')

    ''' RESNET 50 ENSEMBLE model'''

    w_dir = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/weights/only_resnet50/'
    w_paths = {0: f'{w_dir}0.06-1.48.hdf5', 90: f'{w_dir}90.06-1.54.hdf5', 180: f'{w_dir}180.06-1.54.hdf5',
               270: f'{w_dir}270.04-2.02.hdf5'}

    ensemble_model = ensemble_model(RESNET50, w_paths)
    accuracies = topk_accuracy(ensemble_model, [1, 3, 5])
    print_accuracy(accuracies, 'RESNET50 ensemble model')
