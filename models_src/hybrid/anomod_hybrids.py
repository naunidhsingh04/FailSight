"""The five AnoMod hybrids. A hybrid is NOT one combined function or class: each algorithm is defined on its own in stages.py (cnn_stage, lstm_stage, ...), the first one is
trained and produces variables, and the second one is called with those variables and uses their values. A hybrid below is only that hand-over:

    cnn_vars  = cnn_stage(spec, cfg, data)                 # first algorithm: trains, returns its variables
    lstm_vars = lstm_stage(cnn_vars, spec, cfg, data)      # second algorithm: uses the CNN's values, trains, classifies, validates
"""
from .stages import (cnn_stage, transformer_stage, autoencoder_stage, vae_stage, gnn_stage,
                     lstm_stage, bilstm_stage, transformer_second_stage)


def transformer_bilstm(spec, cfg, data=None):
    """Transformer + BiLSTM.
    First part: a Transformer encoder (self-attention) that looks at the whole sequence of 6 windows at once and rewrites every window in the light of the others.
    Second part: a bidirectional LSTM that reads those rewritten windows in order, forwards and backwards, and keeps a summary.
    Passed between them: a (6 windows x hidden) sequence of attention-enriched window vectors. Then a small classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = transformer_stage(spec, cfg, data)
    second_vars = bilstm_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


transformer_bilstm.two_stage = True


def cnn_lstm(spec, cfg, data=None):
    """CNN + LSTM.
    First part: 1-D convolutions along the 6 windows (with batch normalisation) that pick out short patterns such as a sudden jump between two windows.
    Second part: an LSTM that reads the pattern sequence in time order and keeps a summary.
    Passed between them: a (6 x hidden) sequence of pattern detectors. Then a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = cnn_stage(spec, cfg, data)
    second_vars = lstm_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


cnn_lstm.two_stage = True


def ae_lstm(spec, cfg, data=None):
    """Autoencoder + LSTM.
    First part: an autoencoder that squeezes each window into a short code and learns to rebuild the window from it. How badly it rebuilds is added to the loss as an
    auxiliary signal (weight 0.1), so the code has to keep what describes the window.
    Second part: an LSTM that reads the 6 codes in time order.
    Passed between them: the (6 x hidden) code sequence. Then a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = autoencoder_stage(spec, cfg, data)
    second_vars = lstm_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


ae_lstm.two_stage = True


def vae_transformer(spec, cfg, data=None):
    """VAE + Transformer.
    First part: a variational autoencoder that turns each window into a smooth probabilistic code. Its loss (rebuild error + 0.01 x KL divergence) is added as an auxiliary signal.
    Second part: a Transformer encoder that lets the 6 codes attend to each other.
    Passed between them: the (6 x hidden) sequence of codes (the means at prediction time). Then pooling over the 6 windows and a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = vae_stage(spec, cfg, data)
    second_vars = transformer_second_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


vae_transformer.two_stage = True


def gnn_transformer(spec, cfg, data=None):
    """GNN + Transformer.
    First part: a graph neural network over the components of the system (nodes = groups of features that belong together, edges = components that share a resource or
    work together; see docs/HYBRID_MODELS.md). Per window each node embeds its own numbers and two graph-convolution layers let neighbouring nodes exchange information.
    Second part: a Transformer encoder over the 6 windows.
    Passed between them: one vector per window that summarises the whole graph (6 x hidden). Then pooling and a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = gnn_stage(spec, cfg, data)
    second_vars = transformer_second_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


gnn_transformer.two_stage = True


ANOMOD_HYBRIDS = {"Transformer_BiLSTM": transformer_bilstm, "AE_LSTM": ae_lstm, "GNN_Transformer": gnn_transformer, "CNN_LSTM": cnn_lstm, "VAE_Transformer": vae_transformer}
