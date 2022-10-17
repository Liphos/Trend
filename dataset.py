import numpy as np
import torch
import torchaudio
import librosa
import librosa.display

from keras.datasets import mnist, cifar10

from binreader import open_binary_file
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

from utils.utils import logical_and_arrays, logical_or_arrays

def import_dataset(name:str, split:float=0.2, shuffle=True, extra_args:Dict[str, bool]={}):
    datasets = {"trend":import_data_TREND,
                "noise_trend": import_noise_TREND,
                "mnist": import_mnist,
                "cifar10":import_cifar10,
                "noisy_cifar10":import_noisy_cifar10,
                }
    
    if name in datasets:
        return datasets[name](split, shuffle, extra_args)
    else:
        raise ValueError("This key is not associated with a dataset")
    
def add_impurity(labels:np.ndarray, impurity_level:float, nb_classes:int, impure_class:int=None):
    if impurity_level == 0:
        return labels
    assert np.max(labels) <=1, "this is for biclassification"
    if impure_class is None:
        indicies_impure = np.where((np.random.rand(labels.shape[0])<impurity_level))[0]    
    else:
        indicies_impure = np.where((np.random.rand(labels.shape[0])<impurity_level) & (labels[:, 0]==impure_class))[0]
    if nb_classes == 2:
        labels[indicies_impure] = 1 - labels[indicies_impure]
    else:
        raise ValueError("Not implemented yet")

def import_mnist(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    print(Warning("Split is not supported for MNIST yet"))
    (data_train, labels_train), (data_test, labels_test) = mnist.load_data()
    data_train, data_test = np.expand_dims(data_train, axis=1), np.expand_dims(data_test, axis=1)
    labels_train, labels_test = np.expand_dims(labels_train, axis=-1), np.expand_dims(labels_test, axis=-1)
    max_classes = 10
    if "max_classes" in extra_args:
        max_classes = extra_args["max_classes"]
        
    if max_classes is not None:
        if type(max_classes) == int:
            max_classes_size = max_classes
            if max_classes <10:
                indicies_train = np.where(labels_train<max_classes)[0]
                data_train, labels_train = data_train[indicies_train], labels_train[indicies_train]
                
                indicies_test = np.where(labels_test<max_classes)[0]
                data_test, labels_test = data_test[indicies_test], labels_test[indicies_test]
                
        elif(type(max_classes) == list):
            max_classes_size = len(max_classes)
            indicies_train = [labels_train[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_train[indicies_train[incr]] = incr
            data_train, labels_train = data_train[logical_or_arrays(indicies_train)], labels_train[logical_or_arrays(indicies_train)]
            
            indicies_test = [labels_test[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_test[indicies_test[incr]] = incr
            data_test, labels_test = data_test[logical_or_arrays(indicies_test)], labels_test[logical_or_arrays(indicies_test)]
        else:
            raise TypeError("MaxClasses is not the good type")
        
    #We add noise in the 1 class
    labels_train_impure = np.copy(labels_train)
    if extra_args["impurity_level"]>0:
        add_impurity(labels_train_impure, impurity_level=extra_args["impurity_level"], nb_classes=max_classes_size, impure_class=extra_args["impure_class"] if "impure_class" in extra_args.keys() else None)
    
    labels_train_dict = {'clean': labels_train, 'noisy': labels_train_impure}
    
    return (data_train, labels_train_dict), (data_test, labels_test)
    
def import_cifar10(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    print(Warning("Split is not supported for Cifar yet"))
    (data_train, labels_train), (data_test, labels_test) = cifar10.load_data()
    data_train, data_test = np.swapaxes(np.swapaxes(data_train, 2, 3), 1, 2)/255, np.swapaxes(np.swapaxes(data_test, 2, 3), 1, 2)/255
    
    max_classes = 10
    if "max_classes" in extra_args:
        max_classes = extra_args["max_classes"]
        
    if max_classes is not None:
        if type(max_classes) == int:
            max_classes_size = max_classes
            if max_classes <10:
                indicies_train = np.where(labels_train<max_classes)[0]
                data_train, labels_train = data_train[indicies_train], labels_train[indicies_train]
                
                indicies_test = np.where(labels_test<max_classes)[0]
                data_test, labels_test = data_test[indicies_test], labels_test[indicies_test]
                
        elif(type(max_classes) == list):
            max_classes_size = len(max_classes)
            indicies_train = [labels_train[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_train[indicies_train[incr]] = incr
            data_train, labels_train = data_train[logical_or_arrays(indicies_train)], labels_train[logical_or_arrays(indicies_train)]
            
            indicies_test = [labels_test[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_test[indicies_test[incr]] = incr
            data_test, labels_test = data_test[logical_or_arrays(indicies_test)], labels_test[logical_or_arrays(indicies_test)]
        else:
            raise TypeError("max_classes has not the good type")
        
    #We add noise in the 1 class
    labels_train_impure = np.copy(labels_train)
    if extra_args["impurity_level"]>0:
        add_impurity(labels_train_impure, impurity_level=extra_args["impurity_level"], nb_classes=max_classes_size, impure_class=extra_args["impure_class"] if "impure_class" in extra_args.keys() else None)

    labels_train_dict = {'clean': labels_train, 'noisy': labels_train_impure}
    
    return (data_train, labels_train_dict), (data_test, labels_test)
    
def import_noisy_cifar10(split:float, shuffle:bool, extra_args:Dict[str, bool]): 
    noise_file = torch.load('./data/CIFAR-10_human.pt')
    labels_rand = noise_file['worse_label']
    labels_rand = np.expand_dims(labels_rand, axis=-1)
    
    max_classes = 10
    if "max_classes" in extra_args:
        max_classes = extra_args["max_classes"]
        
    (data_train, labels_train), (data_test, labels_test) = cifar10.load_data()
    
    if max_classes is not None:
        if type(max_classes) == int:
            if max_classes <10:
                indicies_train = np.where((labels_train<max_classes) & (labels_rand<max_classes))[0]
                data_train, new_labels, labels_train = data_train[indicies_train], labels_rand[indicies_train], labels_train[indicies_train]           
                indicies_test = np.where(labels_test<max_classes)[0]
                data_test, labels_test = data_test[indicies_test], labels_test[indicies_test]
                
        elif(type(max_classes) == list):
            indicies_train = [labels_train[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_train[indicies_train[incr]] = incr
            
            indicies_rand = [labels_rand[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_rand[indicies_rand[incr]] = incr
                
            indicies = logical_and_arrays([logical_or_arrays(indicies_train), logical_or_arrays(indicies_rand)])
            data_train, new_labels, labels_train = data_train[indicies], labels_rand[indicies], labels_train[indicies]
            
            indicies_test = [labels_test[:, 0] == elem for elem in max_classes]
            for incr in range(len(max_classes)):
                labels_test[indicies_test[incr]] = incr
            data_test, labels_test = data_test[logical_or_arrays(indicies_test)], labels_test[logical_or_arrays(indicies_test)]
        else:
            raise TypeError("max_classes has not the good type")
    
    data_train, data_test = np.swapaxes(np.swapaxes(data_train, 2, 3), 1, 2)/255, np.swapaxes(np.swapaxes(data_test, 2, 3), 1, 2)/255
    
    labels_train_dict = {'clean': labels_train, 'noisy': new_labels}
    return (data_train, labels_train_dict), (data_test, labels_test)
    
def import_noise_TREND(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    """Import only a file with noise
    """
    data_anthropique = open_binary_file(Path("./data/MLP6_transient_2.bin"))/255
    data_anthropique = np.expand_dims(data_anthropique[:, 256:], axis=1)
    data_anthropique = data_anthropique - np.expand_dims(np.mean(data_anthropique, axis=-1), axis=-1) #We normalize the input
    labels = np.zeros((len(data_anthropique), 1))
    
    return (None, None), (data_anthropique, labels)

def import_data_TREND(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    #Data for signal analysis
    if "mode" in extra_args:
        preprocessing_mode = extra_args["mode"]
    else:
        preprocessing_mode = None
        
    data_selected = open_binary_file(Path("./data/MLP6_selected.bin"))/255
    data_anthropique = open_binary_file(Path("./data/MLP6_transient.bin"))/255
    
    
    if "import_more_noise"  in extra_args and extra_args["import_more_noise"]:
        data_anthropique2 = open_binary_file(Path("./data/MLP6_transient_2.bin"))/255
        data_anthropique3 = open_binary_file(Path("./data/MLP6_transient_3.bin"))/255
        data_anthropique4 = open_binary_file(Path("./data/MLP6_transient_4.bin"))/255
        data_anthropique5 = open_binary_file(Path("./data/MLP6_transient_5.bin"))/255
        data_anthropique = np.concatenate([data_anthropique, data_anthropique2, data_anthropique3, data_anthropique4, data_anthropique5], axis=0)
        
    data_selected = data_selected[:, 256:] #We remove the beginning where there is nothing   
    data_anthropique = data_anthropique[:, 256:]
    
    data_size_signal = len(data_selected)
    data_size_noise = len(data_anthropique)
    print(data_selected.shape)
    print(data_anthropique.shape)
    
    indicies = np.arange(data_size_signal)
    np.random.shuffle(indicies)
    data_selected = data_selected[indicies]
    
    
    indicies = np.arange(data_size_noise)
    np.random.shuffle(indicies)
    data_anthropique = data_anthropique[indicies]
    
    if "import_more_noise"  in extra_args and extra_args["import_more_noise"]: #We put it here because if we shuffle the dataset after concatenating the train and test data will be identical
        data_selected = np.repeat(data_selected, 5, axis=0)
        data_size_signal = len(data_selected)
    
    if preprocessing_mode == "fft":
        data_selected_fft = np.fft.fft(data_selected)
        data_anthropique_fft = np.fft.fft(data_anthropique)
        
        data_selected_fft_r = data_selected_fft.real/np.expand_dims(np.maximum(np.max(data_selected_fft.real, axis=-1), -np.min(data_selected_fft.real, axis=-1)), axis=-1)
        data_selected_fft_i = data_selected_fft.imag/np.expand_dims(np.maximum(np.max(data_selected_fft.imag, axis=-1), -np.min(data_selected_fft.imag, axis=-1)), axis=-1)
        data_selected = np.stack([data_selected, data_selected_fft_r, data_selected_fft_i], axis=1)
        
        data_anthropique_fft_r = data_anthropique_fft.real/np.expand_dims(np.maximum(np.max(data_anthropique_fft.real, axis=-1), -np.min(data_anthropique_fft.real, axis=-1)), axis=-1)
        data_anthropique_fft_i = data_anthropique_fft.imag/np.expand_dims(np.maximum(np.max(data_anthropique_fft.imag, axis=-1), -np.min(data_anthropique_fft.imag, axis=-1)), axis=-1)
        data_anthropique = np.stack([data_anthropique, data_anthropique_fft_r, data_anthropique_fft_i], axis=1)

        data_train = np.concatenate([data_selected[:int(data_size_signal*(1-split))], data_anthropique[:int(data_size_noise*(1-split))]], axis=0)
        data_test = np.concatenate([data_selected[int(data_size_signal*(1-split)):], data_anthropique[int(data_size_noise*(1-split)):]], axis=0)
    
    elif preprocessing_mode == "spectrogram":
        data = np.concatenate([data_selected, data_anthropique], axis=0)
        sgram = librosa.amplitude_to_db(abs(librosa.stft(data, n_fft=80)))
        sgram = np.pad(sgram, [(0,0), (0, 1), (1, 2)])
        sgram = sgram - np.mean(sgram, axis=(-2, -1), keepdims=True)
        sgram = sgram/np.maximum(np.max(sgram, axis=(-2, -1), keepdims=True), -np.min(sgram, axis=(-2, -1), keepdims=True))
                
        data_signal = sgram[:data_size_signal]
        data_noise = sgram[data_size_signal:]
        
        data_train = np.expand_dims(np.concatenate([data_signal[:int(data_size_signal*(1-split))], data_noise[:int(data_size_noise*(1-split))]], axis=0), axis=1)
        data_test = np.expand_dims(np.concatenate([data_signal[int(data_size_signal*(1-split)):], data_noise[int(data_size_noise*(1-split)):]], axis=0), axis=1)
        
        labels_train = np.expand_dims(np.concatenate([np.ones((int(data_size_signal*(1-split)),)), np.zeros((int(data_size_noise*(1-split)),))]), axis=1)
        labels_test = np.expand_dims(np.concatenate([np.ones((int(data_size_signal*(split)),)), np.zeros((int(data_size_noise*(split)),))]), axis=1)
        
        return (data_train, labels_train), (data_test, labels_test)
    
    else:
        data_train = np.expand_dims(np.concatenate([data_selected[:int(data_size_signal*(1-split))], data_anthropique[:int(data_size_noise*(1-split))]]), axis=1)
        data_test = np.expand_dims(np.concatenate([data_selected[int(data_size_signal*(1-split)):], data_anthropique[int(data_size_noise*(1-split)):]]), axis=1)
    
    data_train = data_train - np.expand_dims(np.mean(data_train, axis=-1), axis=-1) #We normalize the input
    data_test = data_test - np.expand_dims(np.mean(data_test, axis=-1), axis=-1)
    
    labels_train = np.expand_dims(np.concatenate([np.ones((int(data_size_signal*(1-split)),)), np.zeros((int(data_size_noise*(1-split)),))]), axis=1)
    labels_test = np.expand_dims(np.concatenate([np.ones((int(data_size_signal*(split)),)), np.zeros((int(data_size_noise*(split)),))]), axis=1)
        
    labels_train_dict = {'clean': labels_train, 'noisy': labels_train}
    return (data_train, labels_train_dict), (data_test, labels_test)