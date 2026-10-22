"""The five LO2 hybrids (same base, same training protocol as the AnoMod ones). One keras.Model each: first component -> second component -> head."""
import numpy as np
import keras
from keras import layers, ops
from .base import HybridBase, SeqTransformer, GraphEncoder, GraphLayer, Head, normalised_adjacency
from .anomod_hybrids import AELSTM


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


class BERTGNN(HybridBase):
    """BERT (frozen text encoder) + GNN.
    First part: a frozen text encoder that turned every distinct log template into a fixed vector before training (BERT-style sentence encoder if available, else TF-IDF + SVD;
    see the spec). A window's template counts scale those vectors.
    Second part: a graph neural network (hand-written graph convolutions) over the template co-occurrence graph (edges = templates that appear together in training windows).
    Passed between them: one vector per template node, scaled by how often the template occurred in the window. The non-template numbers of the window (error rates, volume,
    metrics) go through a small dense layer and join before the head. Only the graph part and the head are trained (the text encoder is frozen by design)."""
    kind = "BERT + GNN"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.tmpl_idx = np.asarray(spec["tmpl_idx"], dtype="int32")
        self.side_idx = np.asarray(spec["side_idx"], dtype="int32")
        self.first = TemplateEncoder(spec["tmpl_emb"])
        self.second = TemplateGNN(spec["tmpl_adj"], d, p)
        self.side = keras.Sequential([layers.Dense(d, activation="relu"), layers.Dropout(p)])
        self.lstm = layers.LSTM(d)
        self.head = Head(spec["C"], d, p)

    def parts(self):
        return {"first": self.first, "second": self.second}

    def call(self, x, training=False):
        nodes = self.first(ops.take(x, self.tmpl_idx, axis=2), training=training)
        g = self.second(nodes, training=training)
        s = self.side(ops.take(x, self.side_idx, axis=2), training=training)
        return self.head(self.lstm(g + s, training=training), training=training)


class TransformerLSTM(HybridBase):
    """Transformer + LSTM.
    First part: a Transformer encoder (self-attention) over the sequence of windows.
    Second part: an LSTM that reads the attention-enriched windows in time order.
    Passed between them: a (T windows x hidden) sequence. Then a classification head. Trained end to end."""
    kind = "Transformer + LSTM"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = SeqTransformer(spec["T"], d, drop=p)
        self.second = layers.LSTM(d)
        self.head = Head(spec["C"], d, p)


class GNNLSTM(HybridBase):
    """GNN + LSTM.
    First part: a graph neural network over the log sources of the system (nodes = the seven service logs, the database log, the load-generator log and the metrics; edges =
    sources whose activity moves together in the training windows, see docs/HYBRID_MODELS.md).
    Second part: an LSTM that reads the per-window graph summaries in time order.
    Passed between them: one vector per window summarising the graph (T x hidden). Then a classification head. Trained end to end."""
    kind = "GNN + LSTM"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        self.first = GraphEncoder(spec["node_idx"], spec["adj"], max(8, d // 2), p)
        self.second = layers.LSTM(d)
        self.head = Head(spec["C"], d, p)


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


class MultimodalTransformerMLP(HybridBase):
    """Multimodal Transformer + MLP.
    First part: two Transformer encoders side by side, a logs branch (log counts, error/warn rates, template counts) and a metrics branch (system metrics), each pooled over the windows.
    Second part: an MLP that fuses the two summaries.
    Passed between them: the two branch summaries joined into one vector (2 x hidden). Then a classification head. Trained end to end."""
    kind = "Multimodal Transformer + MLP"

    def __init__(self, spec, cfg, **kw):
        super().__init__(spec, cfg, **kw)
        d, p = cfg["hidden"], cfg["dropout"]
        li, mi = spec["modal_idx"]
        self.first = MultimodalBranches(spec["T"], d, p, li, mi)
        self.second = keras.Sequential([layers.Dense(2 * d, activation="relu"), layers.BatchNormalization(), layers.Dropout(p), layers.Dense(d, activation="relu")])
        self.head = Head(spec["C"], d, p)


LO2_HYBRIDS = {"BERT_GNN": BERTGNN, "Transformer_LSTM": TransformerLSTM, "AE_LSTM": AELSTM, "GNN_LSTM": GNNLSTM, "Multimodal_Transformer_MLP": MultimodalTransformerMLP}
