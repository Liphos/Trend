import numpy as np
from datasets import load_dataset
from binreader import open_binary_file
from pathlib import Path


def import_dataset(name:str, split:float=0.2):
    datasets = {"minds14":import_minds_hugging_face,
                "trend":import_data_TREND,
                }
    if name in datasets:
        return datasets[name](split)
    else:
        raise ValueError("This key is not associated with a dataset")


def import_minds_hugging_face(split:float):
    minds = load_dataset("PolyAI/minds14", "fr-FR") # for French
    audio_input = np.array([minds["train"][i]["audio"]["array"] for i in range(len(minds["train"]))])
    intent_class = np.array([minds["train"][i]["intent_class"] for i in range(len(minds["train"]))])

    return (minds["train"], minds["test"])

def import_data_TREND(split:float):
    #Data for signal analysis
    data_selected = open_binary_file(Path("./MLP6_selected.bin"))/255
    data_anthropique = open_binary_file(Path("./MLP6_transient.bin"))/255
    data_anthropique2 = open_binary_file(Path("./MLP6_transient_2.bin"))/255
    
    data_size = len(data_selected)
    print(data_selected.shape)
    print(data_anthropique.shape)
    
    data_train = np.concatenate([data_selected[:int(data_size*(1-split))], data_anthropique[:int(data_size*(1-split))]])
    data_test = np.concatenate([data_selected[int(data_size*(1-split)):], data_anthropique[int(data_size*(1-split)):]])
    
    labels_train = np.concatenate([np.ones((int(data_size*(1-split)),)), np.zeros((int(data_size*(1-split)),))])
    labels_test = np.concatenate([np.ones((int(data_size*(split)),)), np.zeros((int(data_size*(split)),))])
    
    indicies = np.arange(len(data_train))
    np.random.shuffle(indicies)
    
    data_train, labels_train = data_train[indicies], labels_train[indicies]
    
    
    indicies = np.arange(len(data_test))
    np.random.shuffle(indicies)
    data_test, labels_test = data_test[indicies], labels_test[indicies]
    
    data_train = data_train[:, 256:] #We remove the beginning where there is nothing
    data_test = data_test[:, 256:]
    
    data_train = data_train - np.expand_dims(np.mean(data_train, axis=-1), axis=-1) #We normalize the input
    data_test = data_test - np.expand_dims(np.mean(data_test, axis=-1), axis=-1)
    
    return (data_train, labels_train), (data_test, labels_test)