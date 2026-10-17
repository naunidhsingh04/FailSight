"""Keras model builders (CNN+BiGRU+Att, CNN+BiLSTM+Att, Fusion, Transformer)."""
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from common import T


@keras.saving.register_keras_serializable(package="ano")
class TemporalAttention(layers.Layer):
    """Additive attention over time. Returns (context, weights)."""

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
        a = tf.nn.softmax(self.score(self.proj(x)), axis=1)  # (b,T,1)
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


def _out(ctx, C, drop):
    x = layers.BatchNormalization()(ctx)
    x = layers.Dense(64, activation="relu", name="embed")(x)
    x = layers.Dropout(drop)(x)
    return layers.Dense(C, activation="softmax", name="out")(x)


def build(name, nfeat, C, units, drop, groups):
    """groups = (n_api, n_logs, n_metrics); feature order: api, logs, metrics, system_is_TT (last column)."""
    inp = keras.Input((T, nfeat))
    att = None
    if name in ("CNN_BiGRU_Att", "CNN_BiLSTM_Att"):
        rnn = layers.GRU if "GRU" in name else layers.LSTM
        x = layers.Conv1D(units, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x); x = layers.Dropout(drop)(x)
        x = layers.Bidirectional(rnn(units, return_sequences=True))(x)
        ctx, att = TemporalAttention(name="attn")(x)
    elif name == "Fusion":
        na, nl, nm = groups
        spans = {"api": range(0, na), "logs": range(na, na + nl), "metrics": range(na + nl, na + nl + nm)}
        sysi = nfeat - 1
        outs = []
        for g, rg in spans.items():
            b = GatherCols(list(rg) + [sysi], name=f"cols_{g}")(inp)
            b = layers.Conv1D(units // 2, 3, padding="same", activation="relu", name=f"conv_{g}")(b)
            b = layers.BatchNormalization()(b); b = layers.Dropout(drop)(b)
            b = layers.Bidirectional(layers.GRU(units // 2, return_sequences=True), name=f"bigru_{g}")(b)
            outs.append(b)
        x = layers.Concatenate()(outs)
        ctx, att = TemporalAttention(name="attn")(x)
    elif name == "Transformer":
        d_model = 64
        x = layers.Dense(d_model)(inp)
        x = PosEnc()(x)
        for _ in range(2):
            h = layers.LayerNormalization()(x)
            h = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(h, h)
            x = x + layers.Dropout(drop)(h)
            h = layers.LayerNormalization()(x)
            h = layers.Dense(units * 2, activation="relu")(h)
            h = layers.Dropout(drop)(h)
            h = layers.Dense(d_model)(h)
            x = x + layers.Dropout(drop)(h)
        x = layers.LayerNormalization()(x)
        ctx = layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(name)
    out = _out(ctx, C, drop)
    model = keras.Model(inp, out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def attn_model(model):
    try:
        return keras.Model(model.input, model.get_layer("attn").output[1])
    except Exception:
        return None


def embed_model(model):
    return keras.Model(model.input, model.get_layer("embed").output)
