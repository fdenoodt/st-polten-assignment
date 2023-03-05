# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision.utils
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset

import random


random.seed(42)

LOG_PATH = "temp/"


def show_image(img, name, save=True):
    print('Original images')
    npimg = img.cpu().detach().numpy()
    # npimg = img.numpy() # old:
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save:
        plt.savefig(f"{LOG_PATH}/gt_img_{name}.png")
    plt.show()


def visualise_output(images, model, device, name, save=True):
    print('Autoencoder reconstruction:')
    with torch.no_grad():
        images = images.to(device)
        images, _, _, _ = model(images)

        images = images.cpu()
        # images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(
            images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        if save:
            plt.savefig(f"{LOG_PATH}/reconstructed_img_{name}.png")
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
        plt.savefig(f"{LOG_PATH}/loss_{name}.png")

    plt.show()


class VanillaVAE(nn.Module):
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py,
    # https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # In: [b, 32, 3, 3]
        # Out: [b, latent_dim, 3, 3]
        self.cnn_mu = nn.Sequential(
            nn.Conv2d(32, latent_dim, 3, stride=1, padding=1),
            nn.ReLU(True))
        
        self.cnn_var = nn.Sequential(
            nn.Conv2d(32, latent_dim, 3, stride=1, padding=1),
            nn.ReLU(True))

        self.flatten = nn.Flatten(start_dim=1)
        self.fc_mu = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )

        # Build Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 3, stride=1, padding=1), #added

            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_cnn(input)  
        # out: [b x 32 x 3 x 3] = b x c x h x w

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.cnn_mu(result) # out: [b x latent_dim x 3 x 3]
        log_var = self.cnn_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # x = self.decoder_lin(z)
        # x = self.unflatten(x)
        x = self.decoder_cnn(z)
        x = torch.sigmoid(x)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu # originally: (batch, latent_dim), now: (batch, latent_dim, 3, 3)

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var) # (batch, latent_dim, 3, 3)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight=0.0025) -> List[Tensor]:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss # shape: (3, 3)
        loss = loss.mean() # shape: (1)
        return [loss, recons_loss.detach(), -kld_loss.detach()]

    def sample(self,
               num_samples: int,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


# inputs, labels = data
# inputs = torch.rand(33, 1, 28, 28)
# inputs = inputs.to('cuda')
# model = VanillaVAE(latent_dim=10, in_channels=1).to('cuda')
# # mu, log_var = model.encode(inputs)

# outputs, inputs, mu, log_var = model(inputs)
# print(f"outputs: {outputs.shape}")
# print(mu.shape)
# print(log_var.shape)


# %%

def train(model: VanillaVAE, train_loader, val_loader, learning_rate, nb_epochs, device):
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    val_losses = []

    for e in range(nb_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            outputs, inputs, mu, log_var = model(inputs)

            loss, _, _ = model.loss_function(outputs, inputs, mu, log_var)

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
                outputs, inputs, mu, log_var = model(inputs)
                loss, _, _ = model.loss_function(outputs, inputs, mu, log_var)
                val_loss += loss.item()

        model.train()
        val_losses.append(val_loss / len(val_loader))

        print(
            f'Latent dimensions: {model.latent_dim} -  Epoch: {e + 1}/{nb_epochs} - Loss: {running_loss / len(train_loader)}')

    # </> end all epochs

    plot_losses(train_losses, val_losses,
                name=f'latent_dims_{model.latent_dim}')
    torch.save(model.state_dict(),
               f'{LOG_PATH}/vae_model_latent_dims_{model.latent_dim}.pt')
    return model


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    BATCH_SIZE = 128 * 2 * 2  # 512
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
        batch_size=BATCH_SIZE,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=mnist_val,
        batch_size=BATCH_SIZE,
        shuffle=False)

    # for LATENT_DIM in [2]:
    for LATENT_DIM in [10]:
        NB_EPOCHS = 10
        learning_rate = 1e-3

        model = VanillaVAE(latent_dim=LATENT_DIM, in_channels=1).to(device)

        model = train(model, train_loader, val_loader,
                      learning_rate=learning_rate, nb_epochs=NB_EPOCHS, device=device)

        model.eval()
        images, labels = next(iter(val_loader))

        # show_image(torchvision.utils.make_grid(
        #     images[1:50], 10, 5), f"latent_dim_{LATENT_DIM}.png")

        visualise_output(images, model, device,
                         f"img_latent_dim_{LATENT_DIM}.png")
