import tensorflow as tf
from data import Data
from tqdm import tqdm
import numpy as np
from models import Patches
import matplotlib.pyplot as plt


path = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'
data = Data(validation_split=0.999, image_size=(256, 256), batch_size=8, seed=123, label_mode='int')
ds = data.load_train('data')
val_data = data.load_val('data')
x_train, y_train = [], []

for images, labels in tqdm(ds.unbatch()):
    x_train.append(images.numpy())
    y_train.append(labels.numpy())



image_size = 256
patch_size = 16
num_patches = (image_size // patch_size) ** 2


plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
