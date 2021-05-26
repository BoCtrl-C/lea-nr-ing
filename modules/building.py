import sys
sys.path.append('.')

from modules.transformers import ARTransformerNum
from modules.utils import NUM_LEVELS


HYPERPARAMETERS = {
    # size "small"
    's': {
        'd_model': NUM_LEVELS,
        'nhead': 2,
        'num_encoder_layers': 6,
        'dim_feedforward': 64
    },
    # size "medium"
    'm': {
        'd_model': 2*NUM_LEVELS,
        'nhead': 2,
        'num_encoder_layers': 8,
        'dim_feedforward': 128
    },
    # size "large"
    'l': {
        'd_model': 4*NUM_LEVELS,
        'nhead': 4,
        'num_encoder_layers': 10,
        'dim_feedforward': 256
    }
}


def build_ARTransformerNum(sz, num_num, num_positions):
    """The function builds the ARTransformerNum model corresponding to the
    provided size (sz).
    """

    if sz not in HYPERPARAMETERS:
        raise RuntimeError('the provided size is not defined')
    
    return ARTransformerNum(
        num_num=num_num,
        num_embeddings=NUM_LEVELS,
        num_positions=num_positions,
        **HYPERPARAMETERS[sz]
    )

def count_parameters(model):
    """The function counts the provided model parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)