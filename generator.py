import torch.nn as nn 
import torch 

class Generator(nn.Module): 
    def __init__(self): 
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.model(x)
    
    