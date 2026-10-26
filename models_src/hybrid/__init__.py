from .anomod_hybrids import ANOMOD_HYBRIDS, transformer_bilstm, cnn_lstm, ae_lstm, vae_transformer, gnn_transformer

REGISTRY = dict(ANOMOD_HYBRIDS)
try:                                    # the LO2 hybrids are added in step 3
    from .lo2_hybrids import LO2_HYBRIDS
    REGISTRY.update(LO2_HYBRIDS)
except ImportError:
    LO2_HYBRIDS = {}
