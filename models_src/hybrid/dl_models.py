"""The four single deep-learning models (family (d)) used on the LO2 slice with the same training protocol as the hybrids: CNN+BiGRU+Attention, CNN+BiLSTM+Attention, Fusion, Transformer.
Each can return its pre-head vector (embed) so an XGBoost head can be fitted on it."""
import keras
from keras import layers, ops
from .base import SeqTransformer, Head


class DLBase(keras.Model):
    kind = ""

    def __init__(self, spec, cfg, **kw):
        super().__init__(**kw)
        self.spec, self.cfg = spec, cfg

    def call(self, x, training=False):
        return self.head(self.features(x, training=training), training=training)

    def embed(self, x):
        return self.features(x, training=False)


class AttentionPool(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.score = layers.Dense(1)

    def call(self, h):
        w = ops.softmax(self.score(h), axis=1)
        self.last_attention = w
        return ops.sum(w * h, axis=1)


class _CNNRNNAtt(DLBase):
    rnn = layers.GRU

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.cnn = keras.Sequential([layers.Conv1D(d, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.Dropout(p)])
        self.seq = layers.Bidirectional(self.rnn(d, return_sequences=True))
        self.pool = AttentionPool()
        self.head = Head(spec["C"], d, p)

    def features(self, x, training=False):
        return self.pool(self.seq(self.cnn(x, training=training), training=training))


class CNNBiGRUAtt(_CNNRNNAtt):
    """CNN + BiGRU + Attention: convolution along the windows, bidirectional GRU, attention pooling over the windows."""
    kind = "CNN + BiGRU + Attention"
    rnn = layers.GRU


class CNNBiLSTMAtt(_CNNRNNAtt):
    """CNN + BiLSTM + Attention: convolution along the windows, bidirectional LSTM, attention pooling over the windows."""
    kind = "CNN + BiLSTM + Attention"
    rnn = layers.LSTM


class Fusion(DLBase):
    """Fusion: a dense branch for the newest window and a recurrent branch for the whole sequence, joined before the head."""
    kind = "Fusion"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.now = keras.Sequential([layers.Dense(d, activation="relu"), layers.BatchNormalization(), layers.Dropout(p)])
        self.hist = layers.GRU(d)
        self.mix = layers.Dense(d, activation="relu")
        self.head = Head(spec["C"], d, p)

    def features(self, x, training=False):
        return self.mix(ops.concatenate([self.now(x[:, -1], training=training), self.hist(x, training=training)], axis=-1))


class TransformerNet(DLBase):
    """Transformer: self-attention encoder over the windows, average pooling."""
    kind = "Transformer"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.enc = SeqTransformer(spec["T"], d, drop=p)
        self.pool = layers.GlobalAveragePooling1D()
        self.head = Head(spec["C"], d, p)

    def features(self, x, training=False):
        return self.pool(self.enc(x, training=training))


DL_MODELS = {"CNN_BiGRU_Att": CNNBiGRUAtt, "CNN_BiLSTM_Att": CNNBiLSTMAtt, "Fusion": Fusion, "Transformer": TransformerNet}
