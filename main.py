from functools import partial
import argparse
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import os
import cleanlab
import numpy as np
from datetime import datetime
from pathlib import Path
import collections
plt.plot([2],[1]) #There is a bug if I don't test pyplot beofre importing torch for some reason
plt.clf()

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import import_dataset, add_impurity
from utils.utils import create_batch_tensorboard, logical_and_arrays, focal_loss
from yml_reader import read_config

def train_epoch(config:Dict[str, Union[str,int, List[int]]], model:torch.nn.Module, training_iter:int, epoch:int, data:np.ndarray, data_labels:np.ndarray, optimizer=None, lr_scheduler=None, criterion:torch.nn=None, writer:SummaryWriter=None, is_testing:bool=False, add_gauss_noise:float=0):
    size = len(data)
    batch_size = config["training"]["batch_size"]
    #We shuffle the dataset
    indicies = np.arange(size)
    np.random.shuffle(indicies)
    data, data_labels = data[indicies], data_labels[indicies]
    if not is_testing:
        data = data + np.random.normal(0, add_gauss_noise, data.shape)
        
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
        if is_testing:
            with torch.no_grad():
                # Make predictions for this batch
                outputs = model(inputs)
                # Compute the loss and its gradients
                loss = criterion(outputs, labels)
        else:        
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            
        accuracy = torch.mean(torch.where(torch.round(outputs)==labels, 1., 0.))
        
        sig_mask = labels==1
        if writer is not None:
            key = "test" if is_testing else "train"
            writer.add_scalar("Loss_"+ key, loss, (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
            writer.add_scalar("Metrics_"+ key +"/Accuracy", accuracy, (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
            writer.add_scalar("Metrics_"+ key +"/TPR", torch.mean(torch.where(torch.round(outputs[sig_mask])==1, 1., 0.)), (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
            writer.add_scalar("Metrics_"+ key +"/TNR", torch.mean(torch.where(torch.round(outputs[~sig_mask])==0, 1., 0.)), (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
            writer.add_scalar("Metrics_"+ key +"/TNR", torch.mean(torch.where(torch.round(outputs[~sig_mask])==0, 1., 0.)), (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], (int(size/batch_size) + 1 ) * epoch + i, new_style=True if i==0 else False)
        
        mean_loss = (mean_loss * mean_counter + loss )/(mean_counter + 1)
        mean_accuracy = (mean_accuracy * mean_counter + accuracy )/(mean_counter + 1)
        mean_counter += 1
        
        if not is_testing:
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            lr_scheduler.step()
            # Gather data and report
            
        if not config["training"]["no_print"] and ((i % 30 == 29) or ((i+1) * batch_size >= size)):
            loss, current = loss.item(), i * batch_size + len(inputs) 
            key = "test" if is_testing else "train"
            print(f"{key}:{epoch}, loss: {mean_loss}, accuracy: {mean_accuracy}  [{current}/{size}]")
            mean_loss = 0
            mean_accuracy = 0
            mean_counter = 0
        
if __name__ == "__main__":
    # Gather arguments from command line
    all_args = argparse.ArgumentParser()
    all_args.add_argument("-c", "--config", required=True, help="path to the config file")
    args = vars(all_args.parse_args())
    
    #Gather configs from config file
    config = read_config(args["config"])

    #Set seed
    np.random.seed(config["seed"])
    print("Using " + config["device"] + " device")

    #Import dataset
    dataset_name = config["dataset"]["name"]
    (data_train, labels_train), (data_test, labels_test) = import_dataset(dataset_name, split=0.2, shuffle=True, extra_args=config["dataset"]["extra_args"])

    print(data_train.shape, labels_train.shape)

    #Define writer and prepare the tensorboard
    try:
        os.makedirs("./Models/"+ dataset_name)
    except FileExistsError:
        print("The Directory already exits")
    except:
        print("Unknown exception")
        
        
    tensorboard_log_dir = "./Models/"+ dataset_name + "/" + config["comment"] + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # + str(len(os.listdir("./Models/"+ dataset_name))) To count the experiments

    #training


    models = []
    if config["training"]["mode"] == "cross_training":
        cross_training = config["training"]["extra_args"]["nb_models"]
        for training_iter in range(cross_training):
            if cross_training != 1 and config["training"]["extra_args"]["shared_data"]<1:
                shared_data = config["training"]["extra_args"]["shared_data"]
                data_train_split = data_train[int((len(data_train) *(1-shared_data)*(training_iter/cross_training))):int((len(data_train) * (shared_data + (1-shared_data) * (training_iter/cross_training))))]
                labels_train_split = labels_train[int((len(labels_train) *(1-shared_data)*(training_iter/cross_training))):int((len(labels_train) * (shared_data + (1-shared_data) * (training_iter/cross_training))))]
            else:
                data_train_split = data_train
                labels_train_split = labels_train
            
            #Create model for training 
            model = config["model"]["name"](**config["model"]["extra_args"]).to(config["device"])

            #Define loss funct and optimizer
            if config["loss"]["name"] == "Focal":
                criterion = partial(focal_loss, reduction='mean', **config["loss"]["extra_args"])
            else:
                raise ValueError("Unknown loss")
            optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["optimizer"]["gamma"])

            writer = SummaryWriter(log_dir=tensorboard_log_dir + "/" + str(training_iter))
            create_batch_tensorboard(tensorboard_log_dir)
            
            train_epoch_initializer = partial(train_epoch, config=config, model=model, training_iter=training_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, writer=writer)
            
            #training
            for epoch in range(config["training"]["num_epochs"]):
                print(f"training_iter: [{training_iter+1}/{cross_training}], epoch: {epoch}, lr: {lr_scheduler.get_last_lr()}")
                train_epoch_initializer(epoch=epoch, data=data_train_split, data_labels=labels_train_split, add_gauss_noise=config["training"]["extra_args"]["add_gauss_noise"])
                train_epoch_initializer(epoch=epoch, data=data_test, data_labels=labels_test, is_testing=True)
                if epoch % 2 == 0:
                    torch.save(model.state_dict(), tensorboard_log_dir + "/checkpoint" + str(epoch) +"_" + str(training_iter) + ".pth")

            models.append(model)
            if writer is not None:
                writer.flush()
                writer.close()  
                
    elif config["training"]["mode"] == "relabelling":
        nb_iterations = config["training"]["extra_args"]["nb_iter"]
        nb_models = config["training"]["extra_args"]["nb_models"]
        
        data_test_tensor = torch.as_tensor(data_test, dtype=torch.float32, device=config["device"])
        labels_test_tensor = torch.as_tensor(labels_test, dtype=torch.float32, device=config["device"])
        
        for training_iter in range(nb_iterations):
            #Create model for training 
            models = []
            for incr_model in range(nb_models):
                model = config["model"]["name"](**config["model"]["extra_args"]).to(config["device"])

                #Define loss funct and optimizer
                if config["loss"]["name"] == "Focal":
                    criterion = partial(focal_loss, reduction='mean', **config["loss"]["extra_args"])
                else:
                    raise ValueError("Unknown loss")
                optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

                writer = SummaryWriter(log_dir=tensorboard_log_dir + "/" + str(training_iter) + "_" + str(incr_model))
                create_batch_tensorboard(tensorboard_log_dir)
                
                train_epoch_initializer = partial(train_epoch, config=config, model=model, training_iter=training_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, writer=writer)
                
                #training
                nb_epochs = config["training"]["num_epochs"]
                for epoch in range(config["training"]["num_epochs"]):
                    train_epoch_initializer(epoch=epoch, data=data_train, data_labels=labels_train)
                    torch.save(model.state_dict(), tensorboard_log_dir + "/checkpoint" + str(training_iter) +"_" + str(incr_model) + ".pth")
                
                pred_probs = model(data_test)
                accuracy = torch.mean(torch.where(torch.round(pred_probs)==labels_test, 1., 0.))
                print(f"training_iter: [{training_iter+1}/{nb_iterations}], model: [{incr_model+1}/{nb_models}], accuracy: {accuracy}")    
                models.append(model)
            
            # Define new label  
            new_labels = torch.as_tensor(labels_train, dtype=torch.float32, device=config["device"])
            data_tensor = torch.as_tensor(data_train, dtype=torch.float32, device=config["device"])
            new_labels[logical_and_arrays([torch.where(model(data_tensor)[:, 0]>=0.7, True, False) for model in models])] = 1
            new_labels[logical_and_arrays([torch.where(model(data_tensor)[:, 0]<=1-0.7, True, False) for model in models])] = 0
            print(f"{np.where(new_labels.cpu().numpy()[:,0]==labels_train[:,0])[0].shape}/{len(labels_train)}")
            
            labels_train = new_labels.cpu().numpy()
            
            print(f"total signal: {np.where(labels_train == 1)[0].shape}/{len(labels_train)/2}")
            print(f"total noise: {np.where(labels_train == 0)[0].shape}/{len(labels_train)/2}")
            
            if writer is not None:
                writer.flush()
                writer.close()  
            
            if dataset_name == "trend":
                (_, _), (noisy_data, noisy_label) = import_dataset("noise_trend")
                noisy_data_tensor = torch.as_tensor(noisy_data, dtype=torch.float32, device=config["device"])
                pred_probs = np.mean([model(noisy_data_tensor).detach().cpu().numpy() for model in models], axis=0)
                pred_probs_rounded = np.mean([np.round(model(noisy_data_tensor).detach().cpu().numpy()) for model in models], axis=0)
                print("perf: ", np.where(np.round(pred_probs[:, 0]) == noisy_label[:, 0])[0].shape[0], len(noisy_label))
                print("perf_rounded: ", np.where(np.round(pred_probs_rounded[:, 0]) == noisy_label[:, 0])[0].shape[0], len(noisy_label))

    elif config["training"]["mode"] == "testing":
        #Create model for training 
        models = []
        for folder in os.listdir(f"./Models/{dataset_name}"):
            if os.path.isdir(f"./Models/{dataset_name}/{folder}") and folder.startswith(config["comment"] + '-'):
                for file in os.listdir(f"./Models/{dataset_name}/{folder}"):
                    if file.startswith("checkpoint40_") and file.endswith(".pth"):
                        model = config["model"]["name"](**config["model"]["extra_args"]).to(config["device"])
                        state_dict = torch.load(Path(f"./Models/{dataset_name}/{folder}/{file}")) 
                        model.load_state_dict(state_dict if type(state_dict) == collections.OrderedDict else state_dict())
                        model.eval()
                        models.append(model)
                        
        print(f"{len(models)} models found")
        data_test = torch.as_tensor(data_test, dtype=torch.float32, device=config["device"])
        labels_test = torch.as_tensor(labels_test, dtype=torch.float32, device=config["device"])
        plt.hist(np.mean([model(data_test[torch.where(labels_test==1)[0]]).detach().cpu().numpy() for model in models], axis=0)[:, 0], bins=30)
        plt.title(f"Histogram {dataset_name} Signal")
        
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(np.mean([model(data_test[torch.where(labels_test==0)[0]]).detach().cpu().numpy() for model in models], axis=0)[:, 0], bins=30)
        plt.title(f"Histogram {dataset_name} Noise")
        
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(np.mean([torch.round(model(data_test[torch.where(labels_test==1)[0]])).detach().cpu().numpy() for model in models], axis=0)[:, 0], bins=30)
        plt.title(f"Histogram {dataset_name} Signal rounded")
        
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(np.mean([torch.round(model(data_test[torch.where(labels_test==0)[0]])).detach().cpu().numpy() for model in models], axis=0)[:, 0], bins=30)
        plt.title(f"Histogram {dataset_name} Noise rounded")
        
        plt.show(block=True)
        
        pred_probs = model(data_test)
        data_test = data_test.detach().cpu().numpy()
        labels_test = labels_test.detach().cpu().numpy()
        indicies = np.where(np.round(pred_probs[np.where(labels_test==1)[0], 0].detach().cpu().numpy())==1)[0]
        #TP
        for i in range(10):
            plt.plot([i for i in range(len(data_test[0, 0]))], data_test[np.where(labels_test==1)[0]][indicies[i], 0])
            plt.title("TP")
            plt.show(block=True)
        #FN
        indicies = np.where(np.round(pred_probs[np.where(labels_test==1)[0], 0].detach().cpu().numpy())==0)[0]
        for i in range(10):
            plt.plot([i for i in range(len(data_test[0, 0]))], data_test[np.where(labels_test==1)[0]][indicies[i], 0])
            plt.title("FN")
            plt.show(block=True)
        #FP
        indicies = np.where(np.round(pred_probs[np.where(labels_test==0)[0], 0].detach().cpu().numpy())==1)[0]
        for i in range(10):
            plt.plot([i for i in range(len(data_test[0, 0]))], data_test[np.where(labels_test==0)[0]][indicies[i], 0])
            plt.title("FP")
            plt.show(block=True)
        #TN
        indicies = np.where(np.round(pred_probs[np.where(labels_test==0)[0], 0].detach().cpu().numpy())==0)[0]
        for i in range(10):
            plt.plot([i for i in range(len(data_test[0, 0]))], data_test[np.where(labels_test==0)[0]][indicies[i], 0])
            plt.title("TN")
            plt.show(block=True)
            
        pred_probs = torch.cat((pred_probs, pred_probs), dim=-1)
        pred_probs[:, 0] = 1 - pred_probs[:, 0]
        
        overlapping_class = cleanlab.dataset.find_overlapping_classes(labels=labels_test[:, 0], pred_probs=pred_probs.detach().cpu().numpy())
        print(overlapping_class.head())

    else:
        raise ValueError('This mode of training does not exist')
