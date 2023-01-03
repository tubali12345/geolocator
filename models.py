import tensorflow as tf
from keras import layers
from keras.layers import Dense, Flatten, Average
from tensorflow import keras
import numpy as np

from utils import Parameters


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def ViT(input_shape: tuple,
        x_train: np.array):
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(Parameters.image_size, Parameters.image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    data_augmentation.layers[0].adapt(x_train)
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(Parameters.patch_size)(augmented)
    encoded_patches = PatchEncoder(Parameters.num_patches, Parameters.projection_dim)(patches)

    for _ in range(Parameters.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=Parameters.num_heads, key_dim=Parameters.projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=Parameters.transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=Parameters.mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(Parameters.num_classes)(features)
    return keras.Model(inputs=inputs, outputs=logits)


def RESNET50(input_shape: tuple):
    model = keras.Sequential()
    resnet50_model = keras.applications.ResNet50(include_top=False,
                                                 input_shape=input_shape,
                                                 pooling='avg', classes=Parameters.num_classes)
    model.add(resnet50_model)
    model.add(Flatten())
    model.add(Dense(Parameters.num_classes, activation='softmax'))
    model.summary()
    return model


def RESNET101(input_shape: tuple):
    model = keras.Sequential()

    resnet101_model = keras.applications.ResNet101(include_top=False, input_shape=input_shape,
                                                   pooling='avg', classes=Parameters.num_classes)
    model.add(resnet101_model)
    model.add(Flatten())
    model.add(Dense(Parameters.num_classes, activation='softmax'))
    model.summary()
    return model


def ensemble_model(create_model,
                   input_shape: tuple,
                   weight_paths: dict):
    def_models = {heading: create_model(input_shape) for heading in [0, 90, 180, 270]}

    for heading in def_models:
        def_models[heading].load_weights(weight_paths[heading])

    models = list(def_models.values())

    model_input = keras.Input(shape=input_shape)
    model_outputs = [model(model_input) for model in models]
    ensemble_output = Average()(model_outputs)
    return keras.Model(inputs=model_input, outputs=ensemble_output, name='ensemble')


def boosted_ensemble_model(create_model,
                           input_shape: tuple,
                           weight_paths: dict):
    def_models = {heading: create_model(input_shape) for heading in [0, 90, 180, 270]}

    for heading in def_models:
        def_models[heading].load_weights(weight_paths[heading])

    models = list(def_models.values())

    model_input = keras.Input(shape=input_shape)
    model_outputs = [model(model_input) for model in models]
    merged = keras.layers.Concatenate(axis=1)(model_outputs)
    perceptron = Dense(1000, activation='relu', input_dim=4)(merged)
    output = Dense(50, activation='softmax')(perceptron)
    return keras.Model(inputs=model_input, outputs=output, name='ensemble')
