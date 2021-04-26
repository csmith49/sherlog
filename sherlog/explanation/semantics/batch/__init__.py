"""Utility functions for manipulating tensors with a batch index.

Most of this functionality likely obsolete when `vmap` becomes stable.
"""

import torch

def batches(tensor):
    """The number of batches in a batched tensor.

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    int
    """
    return tensor.shape[0]

def unbatch(tensor):
    """Converts a batched tensor to a set of tensors.

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    Iterable[Tensor]
    """
    yield from tensor.unbind()

def rebatch(tensors):
    """Converts a set of tensors into a rebatched tensor.

    Parameters
    ----------
    tensors : Iterable[Tensor]

    Returns
    -------
    Tensor
    """
    return torch.stack(list(tensors))

def expand(tensor, batches):
    """Expand a tensor into a batched tensor.

    Parameters
    ----------
    tensor : Tensor

    batches : int

    Returns
    -------
    Tensor
    """
    batch_dims = [batches] + ([1] * tensor.dim())
    return tensor.unsqueeze(0).repeat(*batch_dims)

def batch_map(callable, *args):
    """Maps a function across batches.

    Parameters
    ----------
    callable : Callable[Iterable[Tensor], Tensor]

    args : Iterable[Tensor]

    Returns
    -------
    Tensor
    """
    batch_args = zip(*(unbatch(arg) for arg in args))
    results = (callable(*args) for args in batch_args)
    return rebatch(results)

def sum_collapse(tensor):
    """Reduces a multi-dimensional tensor to a batched tensor of sums along each batch.

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    Tensor
    """
    if tensor.dim() == 1: return tensor
    else: return sum_collapse(tensor.sum(dim=-1))