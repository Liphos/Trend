from pathlib import Path 
import os
import numpy as np
from typing import List, Union
import torch
import torch.nn as F


from typing import Dict

def create_batch_tensorboard(dir:str):
    """Create the batch file to launch the TensorBoard

    Args:
        dir (str): the dir where the tensorboard log is
    """
    port = np.random.randint(26100, 26800)
    with open(dir + "/init.bat", "w+") as f:
        f.write("call activate cea\n")
        f.write(f"cd {os.getcwd()}\n")
        f.write(f'start "" http://localhost:{port}/#scalars\n')
        f.write("tensorboard --logdir " + dir + " --port " + str(port))
    f.close()
    
def logical_and_arrays(arrays:List[Union[np.ndarray, torch.tensor]]) -> Union[np.ndarray, torch.tensor]:
    """Return logical and for a list of arrays.

    Args:
        arrays (List[Union[np.ndarray, torch.tensor]]): The list of arrays
    Returns:
        (Union[np.ndarray, torch.tensor]): The logical and of the list of arrays
    """
    if len(arrays) == 0:
        return arrays
    
    if len(arrays) == 1:
        return arrays[0]
    
    result = arrays[0]
    
    for array in arrays[1:]:
        if type(array) == torch.Tensor:
            result = torch.logical_and(result, array)
        elif type(array) == np.ndarray:
            result = np.logical_and(result, array)
        else:
            raise TypeError("The arrays in the list must be numpy arrays or torch tensor")
        
    return result

def logical_or_arrays(arrays:List[Union[np.ndarray, torch.tensor]]) -> Union[np.ndarray, torch.tensor]:
    """Return logical or for a list of arrays.

    Args:
        arrays (List[Union[np.ndarray, torch.tensor]]): The list of arrays
    Returns:
        (Union[np.ndarray, torch.tensor]): The logical or of the list of arrays
    """
    if len(arrays) == 0:
        return arrays
    
    if len(arrays) == 1:
        return arrays[0]
    
    result = arrays[0]
    
    for array in arrays[1:]:
        if type(array) == torch.Tensor:
            result = torch.logical_or(result, array)
        elif type(array) == np.ndarray:
            result = np.logical_or(result, array)
        else:
            raise TypeError("The arrays in the list must be numpy arrays or torch tensor")
        
    return result

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = inputs
    ce_loss = F.BCELoss(reduction="none")(inputs, targets)
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

