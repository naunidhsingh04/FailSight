"""Unit tests of the hybrid models: output shape, and gradients reach BOTH the first and the second part (end-to-end training)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
import keras  # noqa: E402
from models_src.hybrid import REGISTRY  # noqa: E402

T, F, C = 6, 24, 3
SPEC = dict(T=T, F=F, C=C, node_idx=[list(range(0, 8)), list(range(8, 16)), list(range(16, 24))], adj=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            modal_idx=[list(range(0, 12)), list(range(12, 24))], tmpl_idx=list(range(0, 8)), side_idx=list(range(8, 24)),
            tmpl_emb=np.random.RandomState(3).randn(8, 6).astype("float32"), tmpl_adj=np.roll(np.eye(8, dtype=int), 1, 0) + np.roll(np.eye(8, dtype=int), -1, 0))
CFG = dict(hidden=16, dropout=0.1, wd=1e-4, lr=1e-3)


def build(name):
    return REGISTRY[name](SPEC, CFG)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_shape_and_gradients_reach_second_part(name):
    """the first part is trained on its own and then frozen; the second part is trained on top of its values, so gradients reach the second part only"""
    keras.utils.set_random_seed(42)
    m = build(name)
    x = np.random.RandomState(0).randn(8, T, F).astype("float32")
    y = keras.utils.to_categorical(np.random.RandomState(1).randint(0, C, 8), C)
    p = m(x, training=False)
    assert tuple(p.shape) == (8, C)
    assert np.allclose(np.asarray(p).sum(1), 1, atol=1e-4)
    with tf.GradientTape() as tape:
        out = m(x, training=True)
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out))
    parts = m.parts()
    assert not parts["first"].trainable_weights, "the first part is frozen once it has been trained"
    ws = parts["second"].trainable_weights
    assert ws
    grads = tape.gradient(loss, ws)
    assert any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0 for g in grads), f"no gradient reaches the second part of {name}"


@pytest.mark.parametrize("name", list(REGISTRY))
def test_two_stage_training_hands_first_values_to_second(name):
    """the first algorithm is trained (its weights move) and its values are then used, unchanged, by the second algorithm"""
    keras.utils.set_random_seed(0)
    X = np.random.RandomState(0).randn(24, T, F).astype("float32")
    y = np.random.RandomState(1).randint(0, C, 24)
    calls = []

    def fit(model, X_, y_, Xv_, yv_):
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        model.fit(X_, keras.utils.to_categorical(y_, C), epochs=1, verbose=0)
        calls.append(model.name)
        return {"loss": [0.0]}

    m = REGISTRY[name](SPEC, CFG, data=dict(X=X, y=y, Xv=None, yv=None, fit=fit))
    assert len(calls) == (1 if name == "BERT_GNN" else 2), calls         # the frozen BERT stage has nothing to train
    first = [np.array(w) for w in m.first.weights]
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    m.fit(X, keras.utils.to_categorical(y, C), epochs=1, verbose=0)
    assert all(np.array_equal(a, np.array(b)) for a, b in zip(first, m.first.weights)), "training the second part must not change the first part"


@pytest.mark.parametrize("name", ["AE_LSTM", "VAE_Transformer"])
def test_auxiliary_loss_exists(name):
    m = build(name)
    m(np.zeros((2, T, F), "float32"), training=True)
    assert m.losses, "the reconstruction signal must be an auxiliary loss"


def test_each_hybrid_has_two_named_parts_and_docstring():
    for name, cls in REGISTRY.items():
        assert cls.__doc__ and "First part" in cls.__doc__ and "Second part" in cls.__doc__, name
