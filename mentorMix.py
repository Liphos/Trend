from typing import Dict, List
import torch
import torch.nn as F

import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri

class MentorMix():
    def __init__(self):
        self.loss_p_previous = 0
    
    def mentorMixLoss(self, model:torch.nn.Module, x_i: torch.tensor, y_i:torch.tensor, config:Dict) -> List[torch.Tensor]:
        """
            Return the loss for the mentorMix model

        Args:
            model (torch.nn.Module): the architecture model we cant to train
            x_i (torch.tensor): the inputs
            y_i (torch.tensor): the labels
            loss_p_previous (float): the previous loss
            config (Dict): config

        Returns:
            torch.Tensor: the loss to apply
            torch.Tensor: the output of the model
            float: the previous loss
        """
        batch_size = len(x_i)
        with torch.no_grad():
            output = model(x_i) #1
            loss = F.BCELoss(reduction='none')(output, y_i)[:, 0] #1
            loss_p = config["hyperparameter"]["ema"] * (self.loss_p_previous) + (1 - config["hyperparameter"]["ema"]) * torch.sort(loss, dim=0)[0][int(batch_size*config["hyperparameter"]["gamma_p"])] #For the first iter loss_p_previous is at 0
            loss_diff = loss - loss_p # 2 and 3
            v = torch.where(loss_diff <= 0, 1., 0.) #4
            
        Pv = cat.Categorical(F.functional.softmax(v, dim=0)) #5
        indicies_j = Pv.sample(sample_shape=(batch_size, )) #8
        
        x_j = x_i[indicies_j] #8
        y_j = y_i[indicies_j] #8
        
        Beta = diri.Dirichlet(torch.tensor([config["hyperparameter"]["alpha"] for _ in range(2)])) #9
        lambdas = Beta.sample(sample_shape=(batch_size, )).to(config["device"]) #9
        lambdas_max = torch.max(lambdas, dim=-1)[0]  #9 
        lambdas = v * lambdas_max + (1-v) * (1-lambdas_max) #10
        lambdas = torch.unsqueeze(lambdas, dim=-1)
        shape_for_inputs = (batch_size, 1, ) + tuple([1 for _ in range(len(x_i.shape) - 2)])
        x_tilde = lambdas.view(shape_for_inputs) * x_i + (1 - lambdas.view(shape_for_inputs)) * x_j #11
        y_tilde = lambdas * y_i + (1 - lambdas) * y_j #12
        
        output_tilde = model(x_tilde) #13
        
        loss = F.BCELoss()(output_tilde, y_tilde)
            
        return loss, output, loss_p #16