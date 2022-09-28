import numpy as np
from binreader import open_binary_file
import matplotlib.pyplot as plt
from pathlib import Path


data_selected = open_binary_file(Path("./data/MLP6_selected.bin")) /255
data_anthropique = open_binary_file(Path("./data/MLP6_transient.bin")) /255
print(data_selected.shape)
print(data_anthropique.shape)
plt.plot([i for i in range(data_selected.shape[1])], data_selected[2])
plt.title("Trend Signal", fontsize=24)
plt.xlabel("time (ns)", fontsize=24)
plt.xticks(fontsize = 24)
plt.ylabel("tension (normalized)", fontsize=24)
plt.yticks(fontsize = 24)
plt.figure()

plt.plot([i for i in range(data_anthropique.shape[1])], data_anthropique[2])
plt.title("Trend Anthropogenic noise", fontsize=24)
plt.xlabel("time (ns)", fontsize=24)
plt.xticks(fontsize = 24)
plt.ylabel("tension (normalized)", fontsize=24)
plt.yticks(fontsize = 24)
plt.show(block=True)