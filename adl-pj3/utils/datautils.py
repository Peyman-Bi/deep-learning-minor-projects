import torch, os
import numpy as np
from .helpers import fix_seed
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

def get_dataLoader(dataset, valid_portion, batch_sizes, pretrain=False, pre_seed=None, **kwargs):
    valid_size = int(len(dataset)*valid_portion)
    train_size = len(dataset)-valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    if pretrain and pre_seed:
        new_seed = np.random.randint(pre_seed+100, 2*pre_seed+101)
        fix_seed(new_seed, random_lib=True, numpy_lib=True, torch_lib=True)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_sizes[0], **kwargs
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=False, batch_size=batch_sizes[1], **kwargs
    )
    return train_loader, valid_loader