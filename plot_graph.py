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
