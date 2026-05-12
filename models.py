import torch
import torch.nn as nn
import torch.nn.functional as F

# Create Model — MTD-GAN only
from arch.Ours.networks import MTD_GAN_Method


def get_model(name):
    if name == "MTD_GAN_Method":
        model = MTD_GAN_Method()
    else:
        raise ValueError(f"Unknown model name: '{name}'. Only 'MTD_GAN_Method' is supported.")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)

    return model