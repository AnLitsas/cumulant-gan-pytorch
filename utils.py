from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch
import sys
import os 


def load_dataset(data_name): 
    if data_name == "swiss_roll_2d_with_labels": 
       dataset =  pd.read_csv('./datasets/swiss_roll_2d_with_labels.csv').values[:,:-1]
    elif data_name == "toy_example_gmm8":
        dataset =  pd.read_csv('./datasets/toy_example_gmm8.csv').values
    elif data_name == "toy_example_tmm6": 
        dataset =  pd.read_csv('./dataset/toy_example_tmm6.csv').values

    # Split data - 90% train and 10% test 
    np.random.shuffle(dataset)
    n = len(dataset)
    train_ratio = 0.9
    train_size = int(n * train_ratio)

    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    return train_set, test_set

def create_dataloaders(train_set, test_set, batch_size): 
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_set)
    test_tensor = torch.FloatTensor(test_set)

    # Create DataLoaders
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

