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
import matplotlib.pyplot as plt
import numpy as np


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
