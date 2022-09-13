from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.plot([2],[1]) #There is a bug if I don't test pyplot beofre importing torch for some reason
plt.clf()

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import import_dataset
from model import SimpleSignalModel, SimpleMnisteModel, SimpleCifarModel, ResnetCifarModel
from utils.utils import create_batch_tensorboard, logical_and_array


config = {
    "dataset": {"name":"cifar10",
                "extra_args": {"max_classes": 2, "impurity": 0}},
    "optimizer": {"name": "Adam", "lr":1e-3, "weight_decay":1e-4},
    "training":{"num_epochs":10, "batch_size":64, "cross_training":1},
    "model": ResnetCifarModel,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "comment": "test_convergence"
}


def train_epoch(model:torch.nn.Module, training_iter:int, epoch:int, data:np.ndarray, data_labels:np.ndarray, optimizer, lr_scheduler, criterion:torch.nn, writer:SummaryWriter, is_testing:bool=False):
    size = len(data)
    batch_size = config["training"]["batch_size"]
    #We shuffle the dataset
    indicies = np.arange(size)
    np.random.shuffle(indicies)
    data, data_labels = data[indicies], data_labels[indicies]
    
    if is_testing: #Indicate for dropout and batch norm that the model is training
        model.eval()
    else:
        model.train() 
        
    mean_loss = 0
    mean_accuracy = 0
    mean_counter = 0
    for i in range(int(size/batch_size)+1):
        inputs, labels = data[i*batch_size: np.minimum((i+1)*batch_size, size)], data_labels[i*batch_size: np.minimum((i+1)*batch_size, size)] #We normalize the inputs
        # Every data instance is an input + label pair
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=config["device"])
        labels = torch.as_tensor(labels, dtype=torch.float32, device=config["device"])
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
        writer.add_scalar("Loss_"+ key, loss, (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
        writer.add_scalar("Metrics_"+ key +"/Accuracy", accuracy, (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
        writer.add_scalar("Metrics_"+ key +"/TPR", torch.mean(torch.where(torch.round(outputs[sig_mask])==1, 1., 0.)), (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
        writer.add_scalar("Metrics_"+ key +"/TNR", torch.mean(torch.where(torch.round(outputs[~sig_mask])==0, 1., 0.)), (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
        
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
        
        
print("Using " + config["device"] + " device")

#Import dataset
(data_train, labels_train), (data_test, labels_test) = import_dataset(config["dataset"]["name"], split=0.2, shuffle=True, extra_args=config["dataset"]["extra_args"])

print(data_train.shape, labels_train.shape)

#Define writer and prepare the tensorboard
try:
    os.makedirs("./Models/"+ config["dataset"]["name"])
except FileExistsError:
    print("The Directory already exits")
except:
    print("Unknown exception")
    
    
tensorboard_log_dir = "./Models/"+ config["dataset"]["name"] + "/" + config["comment"] + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # + str(len(os.listdir("./Models/"+ dataset_name))) To count the experiments



#training

cross_training = config["training"]["cross_training"]
models = []
for training_iter in range(cross_training):
    if cross_training != 1:
        data_train_split = data_train[int((len(data_train) *(1-0.8)*(training_iter/cross_training))):int((len(data_train) * (0.8 + (1-0.8) * (training_iter/cross_training))))]
        labels_train_split = labels_train[int((len(labels_train) *(1-0.8)*(training_iter/cross_training))):int((len(labels_train) * (0.8 + (1-0.8) * (training_iter/cross_training))))]
    else:
        data_train_split = data_train
        labels_train_split = labels_train
    
    #Create model for training 
    model = config["model"]().to(config["device"])

    #Define loss funct and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    writer = SummaryWriter(log_dir=tensorboard_log_dir + "/" + str(training_iter))
    create_batch_tensorboard(tensorboard_log_dir)
    
    train_epoch_initializer = partial(train_epoch, model=model, training_iter=training_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, writer=writer)
    
    #training
    for epoch in range(config["training"]["num_epochs"]):
        print(f"training_iter: [{training_iter+1}/{cross_training}], epoch: {epoch}, lr: {lr_scheduler.get_last_lr()}")
        train_epoch_initializer(epoch=epoch, data=data_train_split, data_labels = labels_train_split)
        train_epoch_initializer(epoch=epoch, data=data_test, data_labels=labels_test, is_testing=True)
        if epoch % 5 == 0:
            torch.save(model.state_dict, tensorboard_log_dir + "/checkpoint" + str(epoch) +"_" + str(training_iter) + ".pth")

    models.append(model)


#Testing model
#indicies_impure = np.where(np.random.rand(len(labels_test))<0.2)[0]
#labels_test_impure = np.copy(labels_test)
#labels_test_impure[indicies_impure] = 1 - labels_test[indicies_impure]

new_labels = torch.as_tensor(labels_test, dtype=torch.float32, device=config["device"])
new_labels[logical_and_array([torch.where(model(torch.as_tensor(data_test, dtype=torch.float32, device=config["device"]))>=0.7, True, False) for model in models])] = 1
new_labels[logical_and_array([torch.where(model(torch.as_tensor(data_test, dtype=torch.float32, device=config["device"]))<=1-0.7, True, False) for model in models])] = 0
print(np.where(new_labels.cpu().numpy()[:,0]==labels_test[:,0])[0].shape)

writer.flush()
writer.close()  