import torch.nn as F
import torch

class SimpleSignalModel(F.Module):
    def __init__(self):
        super(SimpleSignalModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv1d(1, 128, kernel_size=15, padding=7)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm1d(128)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv1d(128, 128, kernel_size=7, padding=3)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm1d(128)
        self.layers.append(self.batch_norm2)
        
        self.conv3 = F.Conv1d(128, 128, kernel_size=7, padding=3)
        self.layers.append(self.conv3)
        
        self.batch_norm3 = F.BatchNorm1d(128)
        self.layers.append(self.batch_norm3)
        
        self.conv4 = F.Conv1d(128, 128, kernel_size=7, padding=3)
        self.layers.append(self.conv3)
        
        self.batch_norm4 = F.BatchNorm1d(128)
        self.layers.append(self.batch_norm4)
        
        self.dense1 = F.Linear(12*128, 1024)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(1024, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.1)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.maxpool = F.MaxPool1d(7, stride=4, padding=3)
        self.layers.append(self.maxpool)
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.dropout(self.activation(self.batch_norm1(x)))
        
        x = self.dropout(self.activation(self.activation(self.batch_norm2(self.conv2(x))) + x))
        x = self.maxpool(x)
        
        x = self.dropout(self.activation(self.activation(self.batch_norm3(self.conv3(x))) + x))
        x = self.maxpool(x)
        
        x = self.dropout(self.activation(self.activation(self.batch_norm4(self.conv4(x))) + x))
        #x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.activation(self.dense1(x))
        x = self.sigmoid(self.dense2(x))
        
        return x
    
    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()

class SimpleImageModel(F.Module):
    def __init__(self):
        super(SimpleImageModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv2d(1, 8, kernel_size=5, stride=2)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm2d(8)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv2d(8, 16, kernel_size=3)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm2d(16)
        self.layers.append(self.batch_norm2)
                
        self.conv3 = F.Conv2d(16, 32, kernel_size=3)
        self.layers.append(self.conv3)
        
        self.batch_norm3 = F.BatchNorm2d(32)
        self.layers.append(self.batch_norm3)
        
        self.dense1 = F.Linear(32*64, 512)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(512, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.activation(self.batch_norm1(x)))
        
        x = self.dropout(self.activation(self.batch_norm2(self.conv2(x))))
        
        x = self.dropout(self.activation(self.batch_norm3(self.conv3(x))))
                
        x = self.flatten(x)
        x = self.activation(self.dense1(x))
        x = self.sigmoid(self.dense2(x))
        
        return x
    
    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()
