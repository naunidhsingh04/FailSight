"""The five AnoMod hybrids. Each class is one keras.Model: first component -> second component -> classification head, trained end to end.
(One file per model was the plan; the five small classes share their layers, so they live together here and the LO2 hybrids live in lo2_hybrids.py.)"""
import keras
from keras import layers, ops
from .base import HybridBase, SeqTransformer, GraphEncoder, Head


class TransformerBiLSTM(HybridBase):
    """Transformer + BiLSTM.
    First part: a Transformer encoder (self-attention) that looks at the whole sequence of 6 windows at once and rewrites every window in the light of the others.
    Second part: a bidirectional LSTM that reads those rewritten windows in order, forwards and backwards, and keeps a summary.
    Passed between them: a (6 windows x hidden) sequence of attention-enriched window vectors. Then a small classification head. Trained end to end."""
    kind = "Transformer + BiLSTM"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = SeqTransformer(spec["T"], d, drop=p)
        self.second = layers.Bidirectional(layers.LSTM(d))
        self.head = Head(spec["C"], d, p)


class CNNLSTM(HybridBase):
    """CNN + LSTM.
    First part: 1-D convolutions along the 6 windows (with batch normalisation) that pick out short patterns such as a sudden jump between two windows.
    Second part: an LSTM that reads the pattern sequence in time order and keeps a summary.
    Passed between them: a (6 x hidden) sequence of pattern detectors. Then a classification head. Trained end to end."""
    kind = "CNN + LSTM"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = keras.Sequential([layers.Conv1D(d, 3, padding="same", activation="relu"), layers.BatchNormalization(),
                                       layers.Conv1D(d, 3, padding="same", activation="relu"), layers.Dropout(p)])
        self.second = layers.LSTM(d)
        self.head = Head(spec["C"], d, p)


class Autoencoder(layers.Layer):
    """encoder (F -> d, per window) + decoder (d -> F). Returns the code sequence; the rebuilt windows are kept in self.last_recon"""

    def __init__(self, F, d, drop, **kw):
        super().__init__(**kw)
        self.enc = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dropout(drop), layers.Dense(d, activation="relu")])
        self.dec = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dense(F)])

    def call(self, x, training=False):
        z = self.enc(x, training=training)
        self.last_recon = self.dec(z)
        return z


class AELSTM(HybridBase):
    """Autoencoder + LSTM.
    First part: an autoencoder that squeezes each window into a short code and learns to rebuild the window from it. How badly it rebuilds is added to the loss as an
    auxiliary signal (weight 0.1), so the code has to keep what describes the window.
    Second part: an LSTM that reads the 6 codes in time order.
    Passed between them: the (6 x hidden) code sequence. Then a classification head. Trained end to end."""
    kind = "Autoencoder + LSTM"
    recon_weight = 0.1

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = Autoencoder(spec["F"], d, p)
        self.second = layers.LSTM(d)
        self.head = Head(spec["C"], d, p)

    def call(self, x, training=False):
        z = self.first(x, training=training)
        if training:
            self.add_loss(self.recon_weight * ops.mean(ops.square(x - self.first.last_recon)))
        return self.head(self.second(z, training=training), training=training)


class VAE(layers.Layer):
    """variational autoencoder, per window: the encoder gives a mean and a spread, a code is drawn around the mean (training only), the decoder rebuilds the window"""

    def __init__(self, F, d, drop, **kw):
        super().__init__(**kw)
        self.enc = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dropout(drop)])
        self.mu, self.lv = layers.Dense(d), layers.Dense(d)
        self.dec = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dense(F)])

    def call(self, x, training=False):
        h = self.enc(x, training=training)
        mu, lv = self.mu(h), ops.clip(self.lv(h), -6.0, 4.0)
        z = mu + ops.exp(0.5 * lv) * keras.random.normal(ops.shape(mu)) if training else mu
        self.last_recon, self.last_mu, self.last_lv = self.dec(z), mu, lv
        return z


class VAETransformer(HybridBase):
    """VAE + Transformer.
    First part: a variational autoencoder that turns each window into a smooth probabilistic code. Its loss (rebuild error + 0.01 x KL divergence) is added as an auxiliary signal.
    Second part: a Transformer encoder that lets the 6 codes attend to each other.
    Passed between them: the (6 x hidden) sequence of codes (the means at prediction time). Then pooling over the 6 windows and a classification head. Trained end to end."""
    kind = "VAE + Transformer"
    recon_weight, kl_weight = 0.1, 0.01

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = VAE(spec["F"], d, p)
        self.second = SeqTransformer(spec["T"], d, drop=p)
        self.pool = layers.GlobalAveragePooling1D()
        self.head = Head(spec["C"], d, p)

    def call(self, x, training=False):
        z = self.first(x, training=training)
        if training:
            f = self.first
            kl = -0.5 * ops.mean(1 + f.last_lv - ops.square(f.last_mu) - ops.exp(f.last_lv))
            self.add_loss(self.recon_weight * ops.mean(ops.square(x - f.last_recon)) + self.kl_weight * kl)
        return self.head(self.pool(self.second(z, training=training)), training=training)


class GNNTransformer(HybridBase):
    """GNN + Transformer.
    First part: a graph neural network over the components of the system (nodes = groups of features that belong together, edges = components that share a resource or
    work together; see docs/HYBRID_MODELS.md). Per window each node embeds its own numbers and two graph-convolution layers let neighbouring nodes exchange information.
    Second part: a Transformer encoder over the 6 windows.
    Passed between them: one vector per window that summarises the whole graph (6 x hidden). Then pooling and a classification head. Trained end to end."""
    kind = "GNN + Transformer"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = GraphEncoder(spec["node_idx"], spec["adj"], max(8, d // 2), p)
        self.second = SeqTransformer(spec["T"], d, drop=p)
        self.pool = layers.GlobalAveragePooling1D()
        self.head = Head(spec["C"], d, p)

    def call(self, x, training=False):
        return self.head(self.pool(self.second(self.first(x, training=training), training=training)), training=training)


ANOMOD_HYBRIDS = {"Transformer_BiLSTM": TransformerBiLSTM, "AE_LSTM": AELSTM, "GNN_Transformer": GNNTransformer, "CNN_LSTM": CNNLSTM, "VAE_Transformer": VAETransformer}
