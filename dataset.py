import numpy as np
from datasets import load_dataset
from keras.datasets import mnist, cifar10
from binreader import open_binary_file
from pathlib import Path
from typing import Dict

def import_dataset(name:str, split:float=0.2, shuffle=True, extra_args:Dict[str, bool]={}):
    datasets = {"minds14":import_minds_hugging_face,
                "trend":import_data_TREND,
                "mnist": import_mnist,
                "cifar10":import_cifar10,
                }
    
    if name in datasets:
        return datasets[name](split, shuffle, extra_args)
    else:
        raise ValueError("This key is not associated with a dataset")


def import_minds_hugging_face(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    minds = load_dataset("PolyAI/minds14", "fr-FR") # for French
    audio_input = np.array([minds["train"][i]["audio"]["array"] for i in range(len(minds["train"]))])
    intent_class = np.array([minds["train"][i]["intent_class"] for i in range(len(minds["train"]))])

    return (minds["train"], minds["test"])

def import_mnist(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    print(Warning("Split is not supported for MNIST yet"))
    (data_train, labels_train), (data_test, labels_test) = mnist.load_data()
    
    max_classes = 10
    if "max_classes" in extra_args:
        max_classes = extra_args["max_classes"]
        
    impurity = 0
    if "impurity" in extra_args:
        impurity = extra_args["impurity"]
            
    indicies_train = np.where(labels_train<max_classes)[0]
    data_train, labels_train = np.expand_dims(data_train[indicies_train], axis=1), np.expand_dims(labels_train[indicies_train],axis=-1)
    
    indicies_test = np.where(labels_test<max_classes)[0]
    data_test, labels_test = np.expand_dims(data_test[indicies_test], axis=1), np.expand_dims(labels_test[indicies_test], axis=-1)
    
    indicies_impure = np.where(np.random.rand(len(labels_train))<impurity)[0]
    labels_train[indicies_impure] = 1 - labels_train[indicies_impure]

    return (data_train, labels_train), (data_test, labels_test)

def import_cifar10(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    print(Warning("Split is not supported for Cifar yet"))
    (data_train, labels_train), (data_test, labels_test) = cifar10.load_data()
    data_train, data_test = np.swapaxes(np.swapaxes(data_train, 2, 3), 1, 2)/255, np.swapaxes(np.swapaxes(data_test, 2, 3), 1, 2)/255
    
    max_classes = 10
    if "max_classes" in extra_args:
        max_classes = extra_args["max_classes"]
        
    impurity = 0
    if "impurity" in extra_args:
        impurity = extra_args["impurity"]
        
    indicies_train = np.where(labels_train<max_classes)[0]
    data_train, labels_train = data_train[indicies_train], labels_train[indicies_train]
    
    indicies_test = np.where(labels_test<max_classes)[0]
    data_test, labels_test = data_test[indicies_test], labels_test[indicies_test]
        
    indicies_impure = np.where(np.random.rand(len(labels_train))<impurity)[0]
    labels_train[indicies_impure] = 1 - labels_train[indicies_impure]
    
    return (data_train, labels_train), (data_test, labels_test)
    

def import_data_TREND(split:float, shuffle:bool, extra_args:Dict[str, bool]):
    #Data for signal analysis
    if "use_fourier_transform" in extra_args:
        use_fourier_transform = extra_args["use_fourier_transform"]
    
    data_selected = open_binary_file(Path("./data/MLP6_selected.bin"))/255
    data_anthropique = open_binary_file(Path("./data/MLP6_transient.bin"))/255
    data_anthropique2 = open_binary_file(Path("./data/MLP6_transient_2.bin"))/255
    
    data_selected = data_selected[:, 256:] #We remove the beginning where there is nothing
    data_anthropique = data_anthropique[:, 256:]
    
    data_size = len(data_selected)
    print(data_selected.shape)
    print(data_anthropique.shape)
    
    indicies = np.arange(len(data_selected))
    np.random.shuffle(indicies)
    data_selected = data_selected[indicies]
    
    
    indicies = np.arange(len(data_anthropique))
    np.random.shuffle(indicies)
    data_anthropique = data_anthropique[indicies]
    
    use_fourier_transform = False
    if use_fourier_transform:
        data_selected_fft = np.fft.fft(data_selected)
        data_anthropique_fft = np.fft.fft(data_anthropique)
        
        data_selected_fft_r = data_selected_fft.real/np.expand_dims(np.maximum(np.max(data_selected_fft.real, axis=-1), -np.min(data_selected_fft.real, axis=-1)), axis=-1)
        data_selected_fft_i = data_selected_fft.imag/np.expand_dims(np.maximum(np.max(data_selected_fft.imag, axis=-1), -np.min(data_selected_fft.imag, axis=-1)), axis=-1)
        data_selected = np.stack([data_selected, data_selected_fft_r, data_selected_fft_i], axis=1)
        
        data_anthropique_fft_r = data_anthropique_fft.real/np.expand_dims(np.maximum(np.max(data_anthropique_fft.real, axis=-1), -np.min(data_anthropique_fft.real, axis=-1)), axis=-1)
        data_anthropique_fft_i = data_anthropique_fft.imag/np.expand_dims(np.maximum(np.max(data_anthropique_fft.imag, axis=-1), -np.min(data_anthropique_fft.imag, axis=-1)), axis=-1)
        data_anthropique = np.stack([data_anthropique, data_anthropique_fft_r, data_anthropique_fft_i], axis=1)

        data_train = np.concatenate([data_selected[:int(data_size*(1-split))], data_anthropique[:int(data_size*(1-split))]], axis=0)
        data_test = np.concatenate([data_selected[int(data_size*(1-split)):], data_anthropique[int(data_size*(1-split)):]], axis=0)
        
    else:
        data_train = np.expand_dims(np.concatenate([data_selected[:int(data_size*(1-split))], data_anthropique[:int(data_size*(1-split))]]), axis=1)
        data_test = np.expand_dims(np.concatenate([data_selected[int(data_size*(1-split)):], data_anthropique[int(data_size*(1-split)):]]), axis=1)
    
    data_train = data_train - np.expand_dims(np.mean(data_train, axis=-1), axis=-1) #We normalize the input
    data_test = data_test - np.expand_dims(np.mean(data_test, axis=-1), axis=-1)
    
    labels_train = np.expand_dims(np.concatenate([np.ones((int(data_size*(1-split)),)), np.zeros((int(data_size*(1-split)),))]), axis=1)
    labels_test = np.expand_dims(np.concatenate([np.ones((int(data_size*(split)),)), np.zeros((int(data_size*(split)),))]), axis=1)
        
    
    return (data_train, labels_train), (data_test, labels_test)