import torch.nn as F
import torch

class SimpleModel(F.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = []
        
        self.conv1_sig = F.Conv1d(1, 64, kernel_size=15, padding=7)
        self.layers.append(self.conv1_sig)
        
        self.conv1_fft = F.Conv1d(2, 64, kernel_size=15, padding=7)
        self.layers.append(self.conv1_fft)
        
        self.batch_norm1_sig = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm1_sig)
        
        self.batch_norm1_fft = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm1_fft)
                
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
        x_sig = self.maxpool(self.conv1_sig(x[:, :1, :]))
        x_sig = self.dropout(self.activation(self.batch_norm1_sig(x_sig)))
        
        x_fft = self.maxpool(self.conv1_fft(x[:, 1:, :]))
        x_fft = self.dropout(self.activation(self.batch_norm1_fft(x_fft)))
        
        x = torch.cat([x_sig, x_fft], axis=1)
        
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

