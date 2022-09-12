
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.plot([2],[1]) #There is a bug if I don't test pyplot beofre importing torch for some reason
plt.clf()

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import import_dataset
from model import SimpleModel
from utils import create_batch_tensorboard
import xgboost as xgb

dataset_name = "trend"
# Get cpu or gpu device for training.

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

print(f"Using {device} device")

#Define writer and prepare the tensorboard
try:
    os.makedirs("./Models/"+ dataset_name)
except FileExistsError:
    print("The Directory already exits")
except:
    print("Unknown exception")
    
    
comment = "xgb_model"
tensorboard_log_dir = "./Models/"+ dataset_name + "/" + comment + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # + str(len(os.listdir("./Models/"+ dataset_name))) To count the experiments


writer = SummaryWriter(log_dir=tensorboard_log_dir)
create_batch_tensorboard(tensorboard_log_dir)


#Create model for training 
model = SimpleModel().to(device)

#Define loss funct and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
batch_size = 10

#Import dataset
dataset_args = {"use_fourier_transform":True}
data_train, data_test = import_dataset(dataset_name, split=0.2, shuffle=True, extra_args=dataset_args)

learning_rate_range = np.arange(0.01, 1, 0.05)
for lr in learning_rate_range:
    xgb_classifier = xgb.XGBClassifier(eta=lr)
    xgb_classifier.fit(data_train[0][:, 2, :], data_train[1])
    print("lr", lr, xgb_classifier.score(data_train[0][:, 2, :], data_train[1]))
    print(xgb_classifier.score(data_test[0][:, 2, :], data_test[1]))

print(data_train[0].shape, data_train[1].shape)
for k in range(1):
    print(data_train[1][k])
    #plt.plot([i for i in range(len(data_train[0][k]))], data_train[0][k])
    #plt.show()

def train_epoch(epoch:int, data, data_labels, is_testing:bool=False):
    size = len(data)
    
    #We shuffle the dataset
    indicies = np.arange(len(data))
    np.random.shuffle(indicies)
    data, data_labels = data[indicies], data_labels[indicies]
    
    if is_testing: #Indicate for dropout and batch norm that the model is training
        model.eval()
    else:
        model.train() 
        
    mean_loss = 0
    mean_accuracy = 0
    mean_counter = 0
    for i in range(int(len(data)/batch_size)+1):
        inputs, labels = data[i*batch_size: np.minimum((i+1)*batch_size, size)], data_labels[i*batch_size: np.minimum((i+1)*batch_size, size)] #We normalize the inputs
        # Every data instance is an input + label pair
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        if is_testing:
            with torch.no_grad():
                # Make predictions for this batch
                outputs = model(inputs)
                # Compute the loss and its gradients
                loss = criterion(outputs, labels)
        else:
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            
        accuracy = torch.mean(torch.where(torch.round(outputs)==labels, 1., 0.))
        
        sig_mask = labels==1
        key = "test" if is_testing else "train"
        writer.add_scalar("Loss_"+ key, loss, (int(len(data)/batch_size) + 1 ) * epoch + i)
        writer.add_scalar("Metrics_"+ key +"/Accuracy", accuracy, (int(len(data)/batch_size) + 1 ) * epoch + i)
        writer.add_scalar("Metrics_"+ key +"/TPR", torch.mean(torch.where(torch.round(outputs[sig_mask])==1, 1., 0.)), (int(len(data)/batch_size) + 1 ) * epoch + i)
        writer.add_scalar("Metrics_"+ key +"/TNR", torch.mean(torch.where(torch.round(outputs[~sig_mask])==0, 1., 0.)), (int(len(data)/batch_size) + 1 ) * epoch + i)
        
        mean_loss = (mean_loss * mean_counter + loss )/(mean_counter + 1)
        mean_accuracy = (mean_accuracy * mean_counter + accuracy )/(mean_counter + 1)
        mean_counter += 1
        
        if not is_testing:
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            lr_scheduler.step()
            # Gather data and report
            
        if (i % 30 == 29) or ((i+1) * batch_size >= size):
            loss, current = loss.item(), i * batch_size + len(inputs) 
            key = "test" if is_testing else "train"
            print(f"{key}:{epoch}, loss: {mean_loss}, accuracy: {mean_accuracy}  [{current}/{size}]")
            mean_loss = 0
            mean_accuracy = 0
            mean_counter = 0

#training
nb_epoch = 40 
for i in range(nb_epoch):
    print(f"epoch: {i}, lr: {lr_scheduler.get_last_lr()}")
    train_epoch(i, data_train[0], data_train[1])
    train_epoch(i, data_test[0], data_test[1], is_testing=True)
    if i % 5 == 0:
        torch.save(model.state_dict, tensorboard_log_dir + "/checkpoint" + str(i) + ".pth")
    

writer.flush()
writer.close()