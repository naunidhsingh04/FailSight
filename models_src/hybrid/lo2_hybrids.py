"""The five LO2 hybrids (same protocol as the AnoMod ones). As in anomod_hybrids.py, each algorithm is its own function in stages.py; a hybrid trains the first one and hands
its variables to the second one."""
from .stages import bert_stage, template_gnn_stage, transformer_stage, lstm_stage, gnn_stage, multimodal_stage, mlp_stage
from .anomod_hybrids import ae_lstm


def bert_gnn(spec, cfg, data=None):
    """BERT (frozen text encoder) + GNN.
    First part: a frozen text encoder that turned every distinct log template into a fixed vector before training (BERT-style sentence encoder if available, else TF-IDF + SVD;
    see the spec). A window's template counts scale those vectors.
    Second part: a graph neural network (hand-written graph convolutions) over the template co-occurrence graph (edges = templates that appear together in training windows).
    Passed between them: one vector per template node, scaled by how often the template occurred in the window. The non-template numbers of the window (error rates, volume,
    metrics) go through a small dense layer and join before the LSTM and the head. Only the graph part, the LSTM and the head are trained (the text encoder is frozen by design)."""
    first_vars = bert_stage(spec, cfg, data)
    second_vars = template_gnn_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


bert_gnn.two_stage = True


def transformer_lstm(spec, cfg, data=None):
    """Transformer + LSTM.
    First part: a Transformer encoder (self-attention) over the sequence of windows.
    Second part: an LSTM that reads the attention-enriched windows in time order.
    Passed between them: a (T windows x hidden) sequence. Then a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = transformer_stage(spec, cfg, data)
    second_vars = lstm_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


transformer_lstm.two_stage = True


def gnn_lstm(spec, cfg, data=None):
    """GNN + LSTM.
    First part: a graph neural network over the log sources of the system (nodes = the seven service logs, the database log, the load-generator log and the metrics; edges =
    sources whose activity moves together in the training windows, see docs/HYBRID_MODELS.md).
    Second part: an LSTM that reads the per-window graph summaries in time order.
    Passed between them: one vector per window summarising the graph (T x hidden). Then a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = gnn_stage(spec, cfg, data)
    second_vars = lstm_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


gnn_lstm.two_stage = True


def multimodal_transformer_mlp(spec, cfg, data=None):
    """Multimodal Transformer + MLP.
    First part: two Transformer encoders side by side, a logs branch (log counts, error/warn rates, template counts) and a metrics branch (system metrics), each pooled over the windows.
    Second part: an MLP that fuses the two summaries.
    Passed between them: the two branch summaries joined into one vector (2 x hidden). Then a classification head. Trained in two steps: the first part on its own, then the second part on the first part's values."""
    first_vars = multimodal_stage(spec, cfg, data)
    second_vars = mlp_stage(first_vars, spec, cfg, data)
    return second_vars["model"]


multimodal_transformer_mlp.two_stage = True


LO2_HYBRIDS = {"BERT_GNN": bert_gnn, "Transformer_LSTM": transformer_lstm, "AE_LSTM": ae_lstm, "GNN_LSTM": gnn_lstm, "Multimodal_Transformer_MLP": multimodal_transformer_mlp}
