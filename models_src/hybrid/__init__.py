from .base import HybridBase
from .anomod_hybrids import ANOMOD_HYBRIDS, TransformerBiLSTM, CNNLSTM, AELSTM, VAETransformer, GNNTransformer

REGISTRY = dict(ANOMOD_HYBRIDS)
try:                                    # the LO2 hybrids are added in step 3
    from .lo2_hybrids import LO2_HYBRIDS
    REGISTRY.update(LO2_HYBRIDS)
except ImportError:
    LO2_HYBRIDS = {}
