"""The building blocks of the ten hybrids, each defined on its own as a function: it takes the input variable (a tensor), applies that one component and returns the output
variable. A hybrid (anomod_hybrids.py, lo2_hybrids.py) is only a few lines that call one component function, then hand its output variable to the next component function,
for example   cnn_out = cnn(inp, d, p);   lstm_out = lstm(cnn_out, d).   No component is combined with another one here."""
import numpy as np
import keras
from keras import layers, ops
from .base import SeqTransformer, GraphEncoder, GraphLayer, normalised_adjacency


# ---------------------------------------------------------------- sequence components
def cnn(x, d, p, name="cnn"):
    """CNN: two 1-D convolutions along the windows with batch normalisation, picks out short patterns such as a sudden jump between two windows. (B,T,F) -> (B,T,d)"""
    return keras.Sequential([layers.Conv1D(d, 3, padding="same", activation="relu"), layers.BatchNormalization(),
                             layers.Conv1D(d, 3, padding="same", activation="relu"), layers.Dropout(p)], name=name)(x)


def lstm(x, d, name="lstm"):
    """LSTM: reads a (B,T,d) sequence in time order and keeps the last summary. (B,T,d) -> (B,d)"""
    return layers.LSTM(d, name=name)(x)


def bilstm(x, d, name="bilstm"):
    """BiLSTM: an LSTM read forwards and another read backwards, joined. (B,T,d) -> (B,2d)"""
    return layers.Bidirectional(layers.LSTM(d), name=name)(x)


def transformer(x, T, d, p, name="transformer"):
    """Transformer encoder (self-attention) over the windows. (B,T,F) -> (B,T,d)"""
    return SeqTransformer(T, d, drop=p, name=name)(x)


def pool_windows(x, name="pool"):
    """average over the windows. (B,T,d) -> (B,d)"""
    return layers.GlobalAveragePooling1D(name=name)(x)


def mlp(x, d, p, name="mlp"):
    """MLP that fuses a joined vector. (B,2d) -> (B,d)"""
    return keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.BatchNormalization(), layers.Dropout(p), layers.Dense(d, activation="relu")], name=name)(x)


# ---------------------------------------------------------------- autoencoders
class Autoencoder(layers.Layer):
    """encoder (F -> d, per window) + decoder (d -> F). Returns the code sequence; the rebuilt windows are kept in self.last_recon"""

    recon_weight = 0.1

    def __init__(self, F, d, drop, **kw):
        super().__init__(**kw)
        self.enc = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dropout(drop), layers.Dense(d, activation="relu")])
        self.dec = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.Dense(F)])

    def call(self, x, training=False):
        z = self.enc(x, training=training)
        self.last_recon = self.dec(z)
        if training:                                    # auxiliary signal: how badly the window is rebuilt
            self.add_loss(self.recon_weight * ops.mean(ops.square(x - self.last_recon)))
        return z


class VAE(layers.Layer):
    """variational autoencoder, per window: the encoder gives a mean and a spread, a code is drawn around the mean (training only), the decoder rebuilds the window"""

    recon_weight, kl_weight = 0.1, 0.01

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
        if training:                                    # auxiliary signal: rebuild error + KL divergence
            kl = -0.5 * ops.mean(1 + lv - ops.square(mu) - ops.exp(lv))
            self.add_loss(self.recon_weight * ops.mean(ops.square(x - self.last_recon)) + self.kl_weight * kl)
        return z


def autoencoder(x, F, d, p, name="autoencoder"):
    """Autoencoder: each window -> short code (and a rebuilt window kept in .last_recon for the auxiliary loss). (B,T,F) -> (B,T,d)"""
    return Autoencoder(F, d, p, name=name)(x)


def vae(x, F, d, p, name="vae"):
    """VAE: each window -> probabilistic code (mean at prediction time); .last_recon/.last_mu/.last_lv feed the auxiliary loss. (B,T,F) -> (B,T,d)"""
    return VAE(F, d, p, name=name)(x)


# ---------------------------------------------------------------- graph components
def gnn(x, node_idx, adj, d, p, name="gnn"):
    """GNN over the components of the system (feature groups as nodes, two graph-convolution layers). (B,T,F) -> (B,T,d)"""
    return GraphEncoder(node_idx, adj, max(8, d // 2), p, name=name)(x)


class TemplateEncoder(layers.Layer):
    """FROZEN text encoder output: one fixed vector per log template (a small pretrained BERT-style sentence encoder when it could be downloaded, otherwise TF-IDF + SVD;
    which one was used is stored in the spec). It is computed once, before training, and never changes. Input (B,T,K) template counts -> (B,T,K,e) node features."""

    def __init__(self, emb, **kw):
        super().__init__(**kw)
        self.emb = ops.convert_to_tensor(np.asarray(emb, dtype="float32"))        # (K, e), constant (not a weight)

    def call(self, counts, training=False):
        return counts[..., None] * self.emb[None, None]


class TemplateGNN(layers.Layer):
    """two hand-written graph-convolution layers over the template co-occurrence graph, then pooling of the K nodes weighted by how active each template is"""

    def __init__(self, adj, d, drop, **kw):
        super().__init__(**kw)
        a = normalised_adjacency(adj)
        self.g1, self.g2 = GraphLayer(a, d), GraphLayer(a, d)
        self.do, self.d = layers.Dropout(drop), d
        self.att = layers.Dense(1)

    def call(self, h, training=False):
        B, T, K, e = ops.shape(h)[0], ops.shape(h)[1], ops.shape(h)[2], ops.shape(h)[3]
        x = ops.reshape(h, (-1, K, e))
        x = self.g2(self.do(self.g1(x), training=training))
        w = ops.softmax(self.att(x), axis=1)                    # attention over the template nodes (can be inspected)
        self.last_attention = w
        return ops.reshape(ops.sum(w * x, axis=1), (-1, T, self.d))


def bert(x, template_embeddings, name="bert"):
    """BERT (frozen text encoder): a fixed vector per log template, scaled by the template's count in the window. (B,T,K) -> (B,T,K,e)"""
    return TemplateEncoder(template_embeddings, name=name)(x)


def template_gnn(x, adj, d, p, name="template_gnn"):
    """GNN over the template co-occurrence graph. (B,T,K,e) -> (B,T,d)"""
    return TemplateGNN(adj, d, p, name=name)(x)


# ---------------------------------------------------------------- multimodal
class MultimodalBranches(layers.Layer):
    """two Transformer encoders, one for the log features and one for the metric features; each is pooled over the windows and the two summaries are joined"""

    def __init__(self, T, d, p, log_idx, met_idx, **kw):
        super().__init__(**kw)
        self.log_idx, self.met_idx = np.asarray(log_idx, dtype="int32"), np.asarray(met_idx, dtype="int32")
        self.t_log, self.t_met = SeqTransformer(T, d, drop=p), SeqTransformer(T, d, drop=p)
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        a = self.pool(self.t_log(ops.take(x, self.log_idx, axis=2), training=training))
        b = self.pool(self.t_met(ops.take(x, self.met_idx, axis=2), training=training))
        return ops.concatenate([a, b], axis=-1)


def multimodal_transformer(x, T, d, p, log_idx, met_idx, name="multimodal_transformer"):
    """Multimodal Transformer: a logs Transformer and a metrics Transformer side by side, each pooled, then joined. (B,T,F) -> (B,2d)"""
    return MultimodalBranches(T, d, p, log_idx, met_idx, name=name)(x)


# ---------------------------------------------------------------- wiring helpers (they do not belong to any component)
def classifier(x, C, d, p, name="head"):
    """small classification head. (B,d) -> (B,C) class probabilities"""
    from .base import Head
    return Head(C, d, p, name=name)(x)
