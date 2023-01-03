from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import Config


class Data:
    states = {i: state.name for i, state in enumerate(Path(f'{Config.PATH}data').glob('*/'))}

    def __init__(self,
                 validation_split: float = 0.15,
                 image_size: tuple = (256, 256),
                 batch_size: int = 32,
                 seed: int = 123,
                 label_mode: str = 'categorical'):

        self.val_split = validation_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.label_mode = label_mode

    def load_train(self,
                   name: str,
                   path: str = Config.PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="training",
            seed=self.seed,
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

    def load_val(self,
                 name: str,
                 path: str = Config.PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="validation",
            seed=self.seed,
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

    def load_test(self,
                  path: str = Config.PATH) -> tuple:
        x_test, y_test = [], []
        test_ds = keras.preprocessing.image_dataset_from_directory(
            f'{path}test_data',
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

        for images, labels in tqdm(test_ds.unbatch()):
            x_test.append(images.numpy())
            y_test.append(labels.numpy())

        return np.array(x_test), np.array(y_test)


def preprocess_data_sep_by_heading():
    path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/geolocator'
    headings = [0, 90, 180, 270]
    data = Path(f'{path}/data')

    for heading in headings:
        outdir = Path(f'{path}/pictures_{str(heading)}')
        outdir.mkdir(exist_ok=True, parents=True)
        for directory in tqdm(data.glob("*")):
            outdir2 = Path(f'pictures_{str(heading)}/{directory.name}')
            outdir2.mkdir(exist_ok=True, parents=True)
            for file in directory.glob(f"*_{str(heading)}.jpg"):
                file.rename(outdir2 / file.name)