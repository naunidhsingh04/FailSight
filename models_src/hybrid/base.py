"""Shared pieces of the ten hybrid deep-learning models (TensorFlow / Keras, no extra libraries).

Every hybrid is ONE keras.Model subclass. Its code builds a first component and a second component as sub-models, and its forward pass calls the first, feeds the output into the
second and then applies a small classification head. Both parts are trained together end to end (one loss, one optimiser). Exception: BERT+GNN, where the text encoder is frozen
(it only turns each log template into a fixed vector once, before training) and the graph part consumes those vectors.

spec  (what the data looks like):  T windows per sequence, F features, C classes, node_idx (list of feature-index lists, one per graph node), adj (N x N 0/1 matrix), modalities
cfg   (the settings):  hidden, dropout, wd (weight decay), lr
"""
import numpy as np
import keras
from keras import layers, ops

SEED = 42


class EncoderBlock(layers.Layer):
    """one Transformer encoder block: self-attention + feed-forward, each with a residual connection and layer normalisation"""

    def __init__(self, d, heads=2, ff=None, drop=0.2, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=max(4, d // heads))
        self.n1, self.n2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.f1, self.f2 = layers.Dense(ff or 2 * d, activation="relu"), layers.Dense(d)
        self.do = layers.Dropout(drop)

    def call(self, x, training=False):
        a = self.att(x, x, training=training)
        x = self.n1(x + self.do(a, training=training))
        return self.n2(x + self.do(self.f2(self.f1(x)), training=training))


class SeqTransformer(layers.Layer):
    """projects each time step to d numbers, adds a learned position, applies `blocks` encoder blocks. (B,T,F) -> (B,T,d)"""

    def __init__(self, T, d, heads=2, blocks=1, drop=0.2, **kw):
        super().__init__(**kw)
        self.proj = layers.Dense(d)
        self.pos = layers.Embedding(T, d)
        self.blocks = [EncoderBlock(d, heads, drop=drop) for _ in range(blocks)]
        self.T = T

    def call(self, x, training=False):
        h = self.proj(x) + self.pos(ops.arange(self.T))[None]
        for b in self.blocks:
            h = b(h, training=training)
        return h


class GraphLayer(layers.Layer):
    """one graph-convolution layer written by hand: H' = relu( A_hat @ H @ W ), A_hat = D^-1/2 (A + I) D^-1/2 (fixed). H: (B, N, d)"""

    def __init__(self, a_hat, d, **kw):
        super().__init__(**kw)
        self.a_hat = ops.convert_to_tensor(a_hat, dtype="float32")
        self.w = layers.Dense(d)

    def call(self, h):
        return ops.relu(ops.einsum("nm,bmd->bnd", self.a_hat, self.w(h)))


def normalised_adjacency(adj):
    a = (np.asarray(adj, dtype=np.float32) + np.eye(len(adj), dtype=np.float32) > 0).astype(np.float32)
    d = a.sum(1)
    return (a / np.sqrt(np.outer(d, d))).astype(np.float32)


class GraphEncoder(layers.Layer):
    """graph over the components of a system. Per time step: each node embeds its own features (node_idx), two hand-written graph-convolution layers mix neighbouring
    nodes, the node vectors are flattened and projected. (B,T,F) -> (B,T,d)"""

    def __init__(self, node_idx, adj, d, drop=0.2, **kw):
        super().__init__(**kw)
        self.node_idx = [np.asarray(i, dtype="int32") for i in node_idx]
        self.embed = [layers.Dense(d, activation="relu") for _ in self.node_idx]
        a_hat = normalised_adjacency(adj)
        self.g1, self.g2 = GraphLayer(a_hat, d), GraphLayer(a_hat, d)
        self.out = layers.Dense(d, activation="relu")
        self.do = layers.Dropout(drop)
        self.n = len(self.node_idx)
        self.d = d

    def call(self, x, training=False):
        T = ops.shape(x)[1]
        nodes = ops.stack([e(ops.take(x, ix, axis=2)) for e, ix in zip(self.embed, self.node_idx)], axis=2)   # (B,T,N,d)
        h = ops.reshape(nodes, (-1, self.n, self.d))
        h = self.g2(self.do(self.g1(h), training=training)) + h
        return self.out(ops.reshape(h, (-1, T, self.n * self.d)))


class Head(layers.Layer):
    def __init__(self, C, d, drop=0.2, **kw):
        super().__init__(**kw)
        self.seq = keras.Sequential([layers.Dense(d, activation="relu"), layers.Dropout(drop), layers.Dense(C, activation="softmax")])

    def call(self, x, training=False):
        return self.seq(x, training=training)


class HybridBase(keras.Model):
    """Base class. Subclasses set self.first and self.second (and may add an auxiliary loss); call() = head(second(first(x)))."""
    kind = ""

    def __init__(self, spec, cfg, **kw):
        super().__init__(**kw)
        self.spec, self.cfg = spec, cfg

    def parts(self):
        return {"first": self.first, "second": self.second}

    def call(self, x, training=False):
        return self.head(self.second(self.first(x, training=training), training=training), training=training)

    def encode(self, x):
        """output of the first part (what is passed to the second); used for diagrams and tests"""
        return self.first(x, training=False)
