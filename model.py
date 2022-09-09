import torch.nn as F

class SimpleModel(F.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv1d(1, 16, kernel_size=15, padding=7)
        self.layers.append(self.conv1)
                
        self.conv2 = F.Conv1d(16, 32, kernel_size=8, padding=3)
        self.layers.append(self.conv2)
        
        self.conv3 = F.Conv1d(32, 64, kernel_size=8, padding=3)
        self.layers.append(self.conv3)
        

        self.dense1 = F.Linear(383*32, 4096)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(4096, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.5)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.maxpool = F.MaxPool1d(2)
        self.layers.append(self.maxpool)
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.dropout(self.conv1(x)))
        x = self.maxpool(x)
        x = self.activation(self.dropout(self.conv2(x)))
        #x = self.maxpool(x)
        #x = self.activation(self.dropout(self.conv3(x)))
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

