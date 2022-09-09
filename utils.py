from pathlib import Path 
import os
import numpy as np

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