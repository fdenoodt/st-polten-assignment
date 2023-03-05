import time
from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset

import random
random.seed(42)

LOG_PATH = "temp/"


class VanillaVAE(nn.Module):
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]: # input.shape = (512, 1, 28, 28)
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # print("mu", mu.shape)
        # print("log_var", log_var.shape)
        # print("input", input.shape)
        # print("result", result.shape)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

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
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight=1) -> dict:  # 00025
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

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

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
        images = model(images)
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

            loss, recons_loss, kld = model.loss_function(
                outputs, inputs, mu, log_var)

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

        print(
            f'Latent dimensions: {model.latent_dims} -  Epoch: {e + 1}/{nb_epochs} - Loss: {running_loss / len(train_loader)}')

    # </> end all epochs

    plot_losses(train_losses, val_losses,
                name=f'latent_dims_{model.latent_dims}')
    torch.save(model.state_dict(),
               f'{LOG_PATH}/vae_model_latent_dims_{model.latent_dims}.pt')
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

    for LATENT_DIM in [2, 10]:

        NB_EPOCHS = 20
        learning_rate = 1e-3

        model = VanillaVAE(latent_dim=LATENT_DIM, in_channels=1).to(device)

        model = train(model, train_loader, val_loader,
                      learning_rate=learning_rate, nb_epochs=NB_EPOCHS, device=device)

        model.eval()
        images, labels = next(iter(val_loader))

        show_image(torchvision.utils.make_grid(
            images[1:50], 10, 5), f"latent_dim_{LATENT_DIM}.png")

        visualise_output(images, model, device,
                         f"img_latent_dim_{LATENT_DIM}.png")
        time.sleep(5)
