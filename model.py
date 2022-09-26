import torch.nn as F
import torch

class _ResBlock1d(F.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=2, padding:int=None, downsample:bool=False):   
        super().__init__()
        if padding is None:
            padding = int(kernel_size/2)
            
        if downsample:
            self.conv1 = F.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.shortcut = F.Sequential(
                F.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                F.BatchNorm1d(out_channels)
            )
        else:
            self.conv1 = F.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.shortcut = F.Sequential()

        self.conv2 = F.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = F.BatchNorm1d(out_channels)
        self.bn2 = F.BatchNorm1d(out_channels)
        
    def forward(self, input:torch.Tensor):
        shortcut = self.shortcut(input)
        input = F.ReLU()(self.bn1(self.conv1(input)))
        input = F.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return F.ReLU()(input)

class ResNet1d18(F.Module):
    def __init__(self, in_channels:int, resblock:F.Module=_ResBlock1d, kernel_size:int=3, outputs:int=1000, last_activation:str=None):
        super().__init__()
        self.layer0 = F.Sequential(
            F.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7),
            F.MaxPool1d(kernel_size=3, stride=2, padding=1),
            F.BatchNorm1d(16),
            F.ReLU()
        )

        self.layer1 = F.Sequential(
            resblock(16, 16, kernel_size=kernel_size, downsample=False),
            resblock(16, 16, kernel_size=kernel_size, downsample=False)
        )

        self.layer2 = F.Sequential(
            resblock(16, 32, kernel_size=kernel_size, downsample=True),
            resblock(32, 32, kernel_size=kernel_size, downsample=False)
        )

        self.layer3 = F.Sequential(
            resblock(32, 64, kernel_size=kernel_size, downsample=True),
            resblock(64, 64, kernel_size=kernel_size, downsample=False)
        )

        self.layer4 = F.Sequential(
            resblock(64, 128, kernel_size=kernel_size, downsample=True),
            resblock(128, 128, kernel_size=kernel_size, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(128, outputs)
        self.flatten = F.Flatten()
        
        if last_activation == "Sigmoid":
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = torch.nn.Identity()
        
    def forward(self, input):
        input = self.layer0(input)
        #input = self.layer1(input) #We remove it because otherwise the model is to dense
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.flatten(input)
        input = self.fc(input)
        output = self.last_activation(input)
        return output
    
class ResnetImgModel(F.Module):
    def __init__(self):
        super(ResnetImgModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv2d(1, 32, kernel_size=7, padding=1)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm2d(32)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm2d(32)
        self.layers.append(self.batch_norm2)
        
        self.conv3 = F.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.layers.append(self.conv3)
        
        self.batch_norm3 = F.BatchNorm2d(64)
        self.layers.append(self.batch_norm3)
        
        self.dense1 = F.Linear(64*8*8, 512)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(512, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.5)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.maxpool = F.MaxPool2d(3, stride=2)
        self.layers.append(self.maxpool)
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        
        x = self.dropout(self.activation(self.batch_norm1(x)))
        
        x = self.dropout(self.activation(self.activation(self.batch_norm2(self.conv2(x))) + x))
    
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


class SimpleSignalModel(F.Module):
    def __init__(self, last_activation:str=None):
        super(SimpleSignalModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv1d(1, 64, kernel_size=15, padding=7)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv1d(64, 64, kernel_size=7, padding=3)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm2)
        
        self.conv3 = F.Conv1d(64, 64, kernel_size=7, padding=3)
        self.layers.append(self.conv3)
        
        self.batch_norm3 = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm3)
        
        self.conv4 = F.Conv1d(64, 64, kernel_size=7, padding=3)
        self.layers.append(self.conv3)
        
        self.batch_norm4 = F.BatchNorm1d(64)
        self.layers.append(self.batch_norm4)
        
        self.dense1 = F.Linear(12*64, 512)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(512, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.1)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.maxpool = F.MaxPool1d(15, stride=4, padding=7)
        self.layers.append(self.maxpool)
        
        self.flatten = F.Flatten()
        if last_activation == "Sigmoid":
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = torch.nn.Identity()
    
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
        x = self.last_activation(self.dense2(x))
        
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

class SimpleMnistModel(F.Module):
    def __init__(self):
        super(SimpleMnistModel, self).__init__()
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
        
class SimpleCifarModel(F.Module):
    def __init__(self):
        super(SimpleCifarModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm2d(64)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv2d(64, 128, kernel_size=5, padding=1, stride=2)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm2d(128)
        self.layers.append(self.batch_norm2)
        
        self.dense1 = F.Linear(49*128, 2048)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(2048, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.1)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.activation(self.batch_norm1(self.conv1(x))))
        
        x = self.dropout(self.activation(self.batch_norm2(self.conv2(x))))
                        
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
        
class ResnetCifarModel(F.Module):
    def __init__(self):
        super(ResnetCifarModel, self).__init__()
        self.layers = []
        
        self.conv1 = F.Conv2d(3, 32, kernel_size=7, padding=3)
        self.layers.append(self.conv1)
        
        self.batch_norm1 = F.BatchNorm2d(32)
        self.layers.append(self.batch_norm1)
        
        self.conv2 = F.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layers.append(self.conv2)
        
        self.batch_norm2 = F.BatchNorm2d(32)
        self.layers.append(self.batch_norm2)
                
        self.conv3 = F.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layers.append(self.conv3)
        
        self.batch_norm3 = F.BatchNorm2d(64)
        self.layers.append(self.batch_norm3)
        
        self.conv4 = F.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layers.append(self.conv4)
        
        self.batch_norm4 = F.BatchNorm2d(64)
        self.layers.append(self.batch_norm4)
        
        self.dense1 = F.Linear(64*64, 1024)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(1024, 1)
        self.layers.append(self.dense2)
        
        self.dropout = F.Dropout(0.2)
        self.layers.append(self.dropout)
        
        self.activation = F.ReLU()
        self.layers.append(self.activation)
        
        self.maxpool = F.MaxPool2d(3, stride=2)
        self.layers.append(self.maxpool)
        
        self.flatten = F.Flatten()
        self.sigmoid = F.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x_copy = self.dropout(self.activation(self.batch_norm1(x)))
        
        x = self.dropout(self.activation(self.batch_norm2(self.conv2(x_copy))))
        x = self.dropout(self.activation(self.batch_norm3(self.conv3(x))))
        
        x_copy = self.dropout(self.activation(self.batch_norm4(self.conv4(x_copy))))
        
        x = self.activation(x + x_copy) #Inutile pour relu mais on sait jamais
                
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


dict_models = {
    "ResNet1d18": ResNet1d18,
    "ResnetImgModel": ResnetImgModel,
    "SimpleSignalModel": SimpleSignalModel,
    "SimpleMnistModel": SimpleMnistModel,
    "SimpleCifarModel": SimpleCifarModel,
    "ResnetCifarModel": ResnetCifarModel,
}

def model_by_name(model_name:str) -> F.Module:
    """Return model class using the given name

    Args:
        model_name (str): the model name

    Returns:
        F.Module: the class corresponding to the name
    """
    if model_name in dict_models:
        return dict_models[model_name]
    raise ValueError("There is no model with the given name")