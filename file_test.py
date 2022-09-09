import numpy as np
from binreader import open_binary_file
import matplotlib.pyplot as plt
from pathlib import Path


data_selected = open_binary_file(Path("./MLP6_selected.bin"))
data_anthropique = open_binary_file(Path("./MLP6_transient.bin"))
print(data_selected.shape)
print(data_anthropique.shape)
plt.plot([i for i in range(data_selected.shape[1])], data_selected[0])
plt.show()