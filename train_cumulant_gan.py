from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
import discriminator
import numpy as np
import generator
import argparse 
import imageio
import torch 
import utils
import sys  
import os            

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def D_loss(real_logits, fake_logits, beta=0.5, gamma=0.5):
    if beta == 0: 
        d_loss_real = torch.mean(real_logits)
    else: 
        max_val_real = torch.max(-beta * real_logits)
        d_loss_real = -(1.0 / beta) * (torch.log(torch.mean(torch.exp(-beta * real_logits - max_val_real))) + max_val_real)
    
    if gamma == 0: 
        d_loss_fake = torch.mean(fake_logits)
    else: 
        max_val_fake = torch.max(gamma * fake_logits)
        d_loss_fake = (1.0 / gamma) * (torch.log(torch.mean(torch.exp(gamma * fake_logits - max_val_fake))) + max_val_fake)
    
    d_loss = d_loss_real - d_loss_fake
    
    return d_loss

def G_loss(fake_logits, beta=0.5, gamma=0.5):
    if gamma == 0:
        g_loss = - torch.mean(fake_logits)
    else:
        max_val_fake = torch.max(gamma * fake_logits)
        g_loss = - (1.0 / gamma) * (torch.log(torch.mean(torch.exp(gamma * fake_logits - max_val_fake))) + max_val_fake)
    
    return g_loss


def train_model(train_dataloader, generator_model, discriminator_model, optimizer_G, optimizer_D, epochs, result_path, dataset, beta=0.5, gamma=0.5):
    path_to_plots = 'generated_plots' 
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    images = []
    
    for epoch in range(epochs):
        for i, real_data_batch in enumerate(train_dataloader):
            real_data = real_data_batch[0].to(device)

            ##############################
            # Train the Discriminator (D) #
            ##############################

            # Zero the gradients for D
            optimizer_D.zero_grad()

            # Generate fake data
            noise = torch.randn(real_data.shape[0], 8).to(device)
            fake_data = generator_model(noise).detach().to(device)

            # Compute D's outputs for real and fake data
            D_real = discriminator_model(real_data).to(device)
            D_fake = discriminator_model(fake_data).to(device)

            # Compute D's loss
            d_loss_value = - D_loss(D_real, D_fake, beta=beta, gamma=gamma)

            # Update D
            d_loss_value.backward()
            optimizer_D.step()

            if i%5==0:
                ##############################
                # Train the Generator (G)     #
                ##############################

                # Zero the gradients for G
                optimizer_G.zero_grad()

                # Generate fake data again
                noise = torch.randn(real_data.shape[0], 8).to(device)
                fake_data = generator_model(noise).to(device)

                # Compute D's outputs for the fake data
                D_fake = discriminator_model(fake_data).to(device)

                # Compute G's loss
                g_loss_value = G_loss(D_fake, beta=beta, gamma=gamma)

                # Update G
                g_loss_value.backward()
                optimizer_G.step()

        print_interval = epochs//100
        if epoch % print_interval == 0 or epoch == epochs-1 :
            print(f"Epoch {epoch}: D Loss: {d_loss_value.item()} G Loss: {g_loss_value.item()}")
            
            # Generate some data for plotting
            noise = torch.randn(10000, 8).to(device)  # Generate 10000 noise samples
            generated_data = generator_model(noise).detach().cpu().numpy()

            # Create a scatter plot
            plt.scatter(generated_data[:, 0], generated_data[:, 1], s=5)
            plt.title(f'Generated data at epoch {epoch}')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)

            # Save the plot
            image_path = f'{path_to_plots}/epoch_{epoch}.png'
            plt.savefig(image_path)
            plt.close()

            # Store the image path for GIF creation
            images.append(imageio.imread(image_path))
    # Create a GIF
    imageio.mimsave(f'{result_path}/{dataset}_{beta}_{gamma}.gif', images, duration=int(1000/3))
    # Remove all the PNG files in the 'generated_plots' directory
    for filename in os.listdir(path_to_plots):
        if filename.endswith('.png'):
            file_path = os.path.join(path_to_plots, filename)
            os.remove(file_path)


if __name__ == '__main__':
    
    if torch.cuda.is_available():
        # Use GPU
        print("Using GPU")
        device = torch.device("cuda")
    else:
        # Use CPU
        print("Using CPU")
        device = torch.device("cpu")
        
    # Check if the system is Windows or Linux
    if sys.platform == 'win32':
        device = torch.device('cpu')
        print('Windows detected. Using CPU')
        
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-bs', '--batch_size', type=int, default = 1000)
    argparser.add_argument('-e', '--epochs', type=int, default = 100000)
    argparser.add_argument('-lr', '--learning_rate', type=float, default = 0.0001)
    argparser.add_argument('-d', '--dataset', type = str, default = 'swiss_roll_2d_with_labels')
    argparser.add_argument('-b', '--beta', type = float, default = 0)
    argparser.add_argument('-g', '--gamma', type = float, default = 0)
    
    args = argparser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    dataset = args.dataset
    beta = args.beta
    gamma = args.gamma
    
    # Load data and split into train and test sets
    train_set, test_set = utils.load_dataset(dataset)
    
    # Dataloaders
    train_dataloader, test_dataloader = utils.create_dataloaders(train_set, test_set, batch_size)
    
    result_path = './Results'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    

    
    #########################
    # SET UP
    #########################
    generator_model = generator.Generator().to(device)
    discriminator_model = discriminator.Discriminator().to(device)
    
    optimizer_G = torch.optim.Adam(generator_model.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator_model.parameters(), lr=learning_rate)

    train_model(train_dataloader, generator_model, discriminator_model, optimizer_G, optimizer_D, epochs, result_path, dataset, beta, gamma)