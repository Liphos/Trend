import pandas as pd 
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
entity, project = "liphos", "trend"  # set to your entity and project 
names_filter = ['relabel_iter_8_unaminity_trend']

runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list = [], [], []
for run in runs: 
    if run.name in names_filter:
        history = run.history(keys=["Metrics_train/TPR"])
        print(history["_step"].values, history["Metrics_train/TPR"].values)
        
        plt.plot(history["_step"].values, history["Metrics_train/TPR"].values)
        plt.show(block=True)

impurity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
data_mnist = [0.01, 0.03, 0.06, 0.16, 0.39, 0.5, 0.6]
data_cifar = [0.03, 0.08, 0.17, 0.26, 0.41, 0.5, 0.6]

plt.plot(impurity, impurity)
plt.plot(impurity, data_mnist)
plt.plot(impurity, data_cifar)
plt.title("Relabellisation on trend and cifar dataset")
plt.xlabel("impurity at iteration 0")
plt.ylabel("impurity at iteration 5")
plt.legend(["y=x","Mnist", "Cifar"])
plt.show()