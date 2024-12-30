# Copyright (c) CUBOX, Inc. and its affiliates.

import torch

def get_device():
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def clear_cache():
    """Clears CUDA cache."""
    torch.cuda.empty_cache()