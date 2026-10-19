"""Keras custom layers needed to load the saved deep models (same registered names as scripts/models.py) and the embedding helper.
Kept separate so the backend does not import the training code (which creates output folders)."""
import numpy as np
import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="ano")
class TemporalAttention(layers.Layer):
    def __init__(self, dim=32, **kw):
        super().__init__(**kw)
        self.dim = dim
        self.proj = layers.Dense(dim, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, shape):
        self.proj.build(shape)
        self.score.build(tuple(shape[:-1]) + (self.dim,))
        super().build(shape)

    def call(self, x):
        a = tf.nn.softmax(self.score(self.proj(x)), axis=1)
        return tf.reduce_sum(a * x, axis=1), a[..., 0]

    def compute_output_shape(self, shape):
        return (shape[0], shape[2]), (shape[0], shape[1])

    def get_config(self):
        c = super().get_config(); c["dim"] = self.dim; return c


@keras.saving.register_keras_serializable(package="ano")
class GatherCols(layers.Layer):
    def __init__(self, idx, **kw):
        super().__init__(**kw)
        self.idx = [int(i) for i in idx]

    def call(self, x):
        return tf.gather(x, self.idx, axis=-1)

    def compute_output_shape(self, shape):
        return tuple(shape[:-1]) + (len(self.idx),)

    def get_config(self):
        c = super().get_config(); c["idx"] = self.idx; return c


@keras.saving.register_keras_serializable(package="ano")
class PosEnc(layers.Layer):
    def call(self, x):
        t, d = x.shape[1], x.shape[2]
        pos = np.arange(t)[:, None]; i = np.arange(d)[None, :]
        ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
        pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang)).astype("float32")
        return x + pe[None]


def load_keras(path):
    return keras.models.load_model(path, compile=False)


def embedding_model(model):
    return keras.Model(model.input, model.get_layer("embed").output)
