"""One function per algorithm, in the role it plays inside a hybrid. A hybrid has two halves and the work is divided between the two algorithms:

FIRST algorithm  (cnn_stage, transformer_stage, autoencoder_stage, vae_stage, gnn_stage, bert_stage, multimodal_stage)
    turns the raw windows into a learned representation. It is trained on its own (with a small classification head of its own, so the representation is useful), then frozen.
    It returns a dict of VARIABLES: the input, the trained layer, `encode` (a function that gives the representation), its training history.

SECOND algorithm (lstm_stage, bilstm_stage, transformer_second_stage, template_gnn_stage, mlp_stage)
    receives those variables, uses the representation the first algorithm produced as its own input, adds its own layers, and does the final classification and the
    validation (early stopping on the validation fold). It returns the finished hybrid model plus its own variables.

`data` is None when only the architecture is wanted (to load saved weights); otherwise dict(X, y, Xv, yv, fit) where fit(model, X, y, Xv, yv) trains a model."""
import keras
from keras import layers, ops
import numpy as np
from . import components as comp


def raw_input(spec):
    """the input variable: (B, T windows, F features)"""
    return keras.Input((spec["T"], spec["F"]))


# ------------------------------------------------------------------------------------------------ the two roles
def first_stage(name, inp, out, spec, cfg, data):
    """train the first algorithm (when data is given) with its own small classification head, freeze it, and return its variables"""
    d, p = cfg["hidden"], cfg["dropout"]
    encoder = keras.Model(inp, out, name=f"{name}_encoder")
    history = None
    if len(out.shape) <= 3:                                   # (a frozen 4-D text encoder, e.g. BERT, has nothing to train)
        pooled = layers.GlobalAveragePooling1D(name=f"{name}_pool")(out) if len(out.shape) == 3 else out
        own_model = keras.Model(inp, comp.classifier(pooled, spec["C"], d, p, name=f"{name}_head"), name=f"{name}_alone")
        if data is not None:
            history = data["fit"](own_model, data["X"], data["y"], data["Xv"], data["yv"])
    encoder.trainable = False
    return dict(name=name, input=inp, layer=encoder.get_layer(name), encoder=encoder, history=history,
                encode=lambda x: encoder(x, training=False))          # the values the first algorithm produced, for the second one to use


def second_stage(name, inp, out, first, spec, cfg, data):
    """add the final classification head, train (validated) on top of the first algorithm's frozen values, and return the finished hybrid with its variables"""
    d, p = cfg["hidden"], cfg["dropout"]
    model = keras.Model(inp, comp.classifier(out, spec["C"], d, p), name=f"{first['name']}_{name}")
    history = data["fit"](model, data["X"], data["y"], data["Xv"], data["yv"]) if data is not None else None
    model.first, model.second = first["layer"], model.get_layer(name)       # the two named parts (tests, explain scripts, notebook)
    model.parts = lambda: {"first": model.first, "second": model.second}
    return dict(name=name, model=model, history=history, first=first)


# ------------------------------------------------------------------------------------------------ first algorithms
def cnn_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    cnn_out = comp.cnn(inp, cfg["hidden"], cfg["dropout"])
    return first_stage("cnn", inp, cnn_out, spec, cfg, data)


def transformer_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    transformer_out = comp.transformer(inp, spec["T"], cfg["hidden"], cfg["dropout"])
    return first_stage("transformer", inp, transformer_out, spec, cfg, data)


def autoencoder_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    autoencoder_out = comp.autoencoder(inp, spec["F"], cfg["hidden"], cfg["dropout"])         # also learns to rebuild the window (auxiliary loss, weight 0.1)
    return first_stage("autoencoder", inp, autoencoder_out, spec, cfg, data)


def vae_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    vae_out = comp.vae(inp, spec["F"], cfg["hidden"], cfg["dropout"])                          # rebuild error + 0.01 x KL as auxiliary loss
    return first_stage("vae", inp, vae_out, spec, cfg, data)


def gnn_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    gnn_out = comp.gnn(inp, spec["node_idx"], spec["adj"], cfg["hidden"], cfg["dropout"])
    return first_stage("gnn", inp, gnn_out, spec, cfg, data)


def multimodal_stage(spec, cfg, data=None):
    inp = raw_input(spec)
    li, mi = spec["modal_idx"]
    multimodal_out = comp.multimodal_transformer(inp, spec["T"], cfg["hidden"], cfg["dropout"], li, mi)
    return first_stage("multimodal_transformer", inp, multimodal_out, spec, cfg, data)


def bert_stage(spec, cfg, data=None):
    """frozen text encoder: nothing is trained, it only turns the template counts into per-template vectors"""
    inp = raw_input(spec)
    bert_out = comp.bert(ops.take(inp, np.asarray(spec["tmpl_idx"], dtype="int32"), axis=2), spec["tmpl_emb"])
    return first_stage("bert", inp, bert_out, spec, cfg, data)


# ------------------------------------------------------------------------------------------------ second algorithms (each takes the first one's variables)
def lstm_stage(first, spec, cfg, data=None):
    inp = first["input"]
    lstm_out = comp.lstm(first["encode"](inp), cfg["hidden"])
    return second_stage("lstm", inp, lstm_out, first, spec, cfg, data)


def bilstm_stage(first, spec, cfg, data=None):
    inp = first["input"]
    bilstm_out = comp.bilstm(first["encode"](inp), cfg["hidden"])
    return second_stage("bilstm", inp, bilstm_out, first, spec, cfg, data)


def transformer_second_stage(first, spec, cfg, data=None):
    inp = first["input"]
    transformer_out = comp.transformer(first["encode"](inp), spec["T"], cfg["hidden"], cfg["dropout"])
    pooled = comp.pool_windows(transformer_out)
    return second_stage("transformer", inp, pooled, first, spec, cfg, data)


def template_gnn_stage(first, spec, cfg, data=None):
    """graph network over the template vectors the BERT stage produced; the non-template numbers of the window join, and an LSTM reads the result in time order"""
    d, p = cfg["hidden"], cfg["dropout"]
    inp = first["input"]
    gnn_out = comp.template_gnn(first["encode"](inp), spec["tmpl_adj"], d, p)
    side = ops.take(inp, np.asarray(spec["side_idx"], dtype="int32"), axis=2)
    side_out = keras.Sequential([layers.Dense(d, activation="relu"), layers.Dropout(p)], name="side")(side)
    lstm_out = comp.lstm(layers.Add()([gnn_out, side_out]), d)
    return second_stage("template_gnn", inp, lstm_out, first, spec, cfg, data)


def mlp_stage(first, spec, cfg, data=None):
    inp = first["input"]
    mlp_out = comp.mlp(first["encode"](inp), cfg["hidden"], cfg["dropout"])
    return second_stage("mlp", inp, mlp_out, first, spec, cfg, data)
