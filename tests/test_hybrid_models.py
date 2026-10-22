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
def test_shape_and_gradients_reach_both_parts(name):
    keras.utils.set_random_seed(42)
    m = build(name)
    x = np.random.RandomState(0).randn(8, T, F).astype("float32")
    y = keras.utils.to_categorical(np.random.RandomState(1).randint(0, C, 8), C)
    p = m(x, training=False)
    assert tuple(p.shape) == (8, C)
    assert np.allclose(np.asarray(p).sum(1), 1, atol=1e-4)
    with tf.GradientTape() as tape:
        out = m(x, training=True)
        loss = keras.losses.categorical_crossentropy(y, out)
        loss = tf.reduce_mean(loss) + tf.add_n(m.losses) if m.losses else tf.reduce_mean(loss)
    parts = m.parts()
    allw = [w for part in parts.values() for w in part.trainable_weights]
    grads_all = dict(zip([id(w) for w in allw], tape.gradient(loss, allw)))
    for part_name, part in parts.items():
        ws = part.trainable_weights
        if name == "BERT_GNN" and part_name == "first":
            assert not ws, "the BERT-style template encoder is frozen by design"
            continue
        assert ws, part_name
        grads = [grads_all[id(w)] for w in ws]
        assert any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0 for g in grads), f"no gradient reaches the {part_name} part of {name}"


@pytest.mark.parametrize("name", ["AE_LSTM", "VAE_Transformer"])
def test_auxiliary_loss_exists(name):
    m = build(name)
    m(np.zeros((2, T, F), "float32"), training=True)
    assert m.losses, "the reconstruction signal must be an auxiliary loss"


def test_each_hybrid_has_two_named_parts_and_docstring():
    for name, cls in REGISTRY.items():
        assert cls.__doc__ and "First part" in cls.__doc__ and "Second part" in cls.__doc__, name
