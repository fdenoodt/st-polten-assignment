# Your task is to design a convolutional autoencoder in a deep learning framework of your choice (Tensorflow, Pytorch, etc.).

# The goal is to train the autoencoder for the unsupervised task of image reconstruction. In the architecture of the autoencoder it should be
# possible to specify the size of the bottleneck layer (i.e. the dimension of the learned latent space).
# Split the dataset into a training and test set.
# Run two training sessions:
#   first with a dimension of the bottleneck layer of 2 and second with a dimension > 2,
#   determined empirically to show a good tradeoff for image reconstruction.
# Investigate how well the two models work for the task of image reconstruction on independent test data.

# Next, take a number of random samples from the test set of your dataset (e.g. 1000 random samples) and use the two trained models to obtain latent vectors for each test sample.
# Now visualize the latent vectors and show how the classes in the test set distribute over space. For the model with latent space dimension of 2 use a scatter plot. For the model with higher dimension,
# use e.g. a dimensionality reduction technique like UMAP or t-SNE for plotting or some other visualization method of your choice. Can you identify meaningful cluster structures for the test samples in
# the latent space? Do the classes cluster well? How do the two representations learned from the two models compare?


# %%
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import os
from architectures_vae import VariationalAutoencoder

import random
random.seed(42)

log_path = "vae_logs/"


# %%

# code strongly based on:
# https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/autoencoder.ipynb#scrollTo=ztYkaqtAr_VZ


def show_image(img, name, save=True):
    print('Original images')
    npimg = img.cpu().detach().numpy()
    # npimg = img.numpy() # old:
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save:
        plt.savefig(f"{log_path}/gt_img_{name}.png")
    plt.show()



def visualise_output(images, model, device, name, save=True):
    print('Autoencoder reconstruction:')
    with torch.no_grad():
        images = images.to(device)
        images = model(images)
        images = images.cpu()
        # images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(
            images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        if save:
            plt.savefig(f"{log_path}/reconstructed_img_{name}.png")
        plt.show()


def plot_losses(train_loss, val_loss, name, save=True):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, '--', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'Training and validation loss - {name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save:
        plt.savefig(f"{log_path}/loss_{name}.png")

    plt.show()




# %%


def train(model: VariationalAutoencoder, train_loader, val_loader, learning_rate, nb_epochs, device):
    model.train()
    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=1e-5
    #     )

    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    val_losses = []

    for e in range(nb_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            outputs = model(inputs)

            # loss = criterion(outputs, inputs) + model.encoder.kl
            loss = ((inputs - outputs)**2).sum() + model.encoder.kl

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            running_loss += loss.item()
        # </> end single epoch

        train_losses.append(running_loss / len(train_loader))

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                # loss = criterion(outputs, inputs)
                loss = ((inputs - outputs)**2).sum() + model.encoder.kl
                val_loss += loss.item()

        model.train()
        val_losses.append(val_loss / len(val_loader))

        print(f'Latent dimensions: {model.latent_dims} -  Epoch: {e + 1}/{nb_epochs} - Loss: {running_loss / len(train_loader)}')

    # </> end all epochs

    plot_losses(train_losses, val_losses, name=f'latent_dims_{model.latent_dims}')
    torch.save(model.state_dict(), f'{log_path}/vae_model_latent_dims_{model.latent_dims}.pt')
    return model


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_size = 128 * 2 * 2 # 512
    mnist_train = dataset.MNIST(
        "./", train=True,
        transform=transforms.ToTensor(),
        download=True)

    mnist_val = dataset.MNIST(
        "./", train=False,
        transform=transforms.ToTensor(),
        download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=mnist_val,
        batch_size=batch_size,
        shuffle=False)

   

    for latent_dim in [2, 10]:
    # for latent_dim in [2, 5, 10, 20, 50, 100]:

        num_epochs = 20
        learning_rate = 1e-3  # = 0.001

        model = VariationalAutoencoder(latent_dim).to(device)
        model = train(
            model,
            train_loader,
            val_loader,
            learning_rate=learning_rate,
            nb_epochs=num_epochs,
            device=device)

        model.eval()
        images, labels = next(iter(val_loader))
        show_image(torchvision.utils.make_grid(images[1:50], 10, 5), f"latent_dim_{latent_dim}.png")
        
        visualise_output(images, model, device, f"img_latent_dim_{latent_dim}.png")
        import time
        time.sleep(5)

# %%    
