from tensorflow import keras
from pathlib import Path
from tqdm import tqdm


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


class Data:
    PATH = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'

    def __init__(self, validation_split: float, image_size: tuple[int, int], batch_size: int, seed: int):
        self.val_split = validation_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

    def load_train(self, name: str, path=PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="training",
            seed=self.seed,
            label_mode='categorical',
            image_size=self.image_size,
            batch_size=self.batch_size)

    def load_test(self, name: str, path=PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="validation",
            seed=self.seed,
            label_mode='categorical',
            image_size=self.image_size,
            batch_size=self.batch_size)