from pathlib import Path 
import os
import numpy as np
from typing import List

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
    
def logical_and_array(arrays:List[np.ndarray]) -> np.ndarray:
    """Return logical and for a list of arrays.

    Args:
        arrays (List[np.ndarray]): The list of arrays
    Returns:
        (np.ndarray): The logical and of the list of arrays
    """
    if len(arrays) == 1:
        return arrays[0]
    
    result = arrays[0]
    
    for array in arrays[1:]:
        result = np.logical_and(result, array)
        
    return result