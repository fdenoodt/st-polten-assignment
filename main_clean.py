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


# %%

# code strongly based on:
# https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/autoencoder.ipynb#scrollTo=ztYkaqtAr_VZ

def plot_losses(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, '--', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plt.imsave('loss.png', plt.gcf())

    plt.show()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.c = 4

        # Encoder block
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(in_features=self.c*7*7,
                            out_features=latent_dims)  # c*2*7*7

        # Decoder block
        self.fc_dec = nn.Linear(in_features=latent_dims,
                                out_features=self.c * 7 * 7)

        self.trans_conv5 = nn.ConvTranspose2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.trans_conv6 = nn.ConvTranspose2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.trans_conv7 = nn.ConvTranspose2d(
            in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.trans_conv8 = nn.Conv2d(
            in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

    def forward(self, x):  # (b, 1, 28, 28)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        encoded = F.relu(self.conv4(x3))  # (b, 1, 7, 7)

        # flatten batch of multi-channel feature maps to a batch of feature vectors
        encoded = encoded.view(encoded.size(0), -1)
        encoded = self.fc(encoded)

        x = self.fc_dec(encoded)
        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.c, 7, 7)
        x4 = F.relu(self.trans_conv5(x))
        x5 = F.relu(self.trans_conv6(x4))
        x6 = F.relu(self.trans_conv7(x5))

        decoded = F.relu(self.trans_conv8(x6))
        return decoded

# x = torch.randn(1, 1, 28, 28)
# model = Autoencoder()
# print(model(x).shape)


# %%


def train(model, train_loader, val_loader, learning_rate, epoch, device):
    model.train()
    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5)

    train_losses = []
    val_losses = []

    for e in range(epoch):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, inputs)

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
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        model.train()
        val_losses.append(val_loss / len(val_loader))

        print(
            f'Epoch: {e + 1}/{epoch} - Loss: {running_loss / len(train_loader)}')

    # </> end all epochs

    plot_losses(train_losses, val_losses)
    return model


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    latent_dims = 10
    num_epochs = 5
    batch_size = 128
    learning_rate = 1e-3  # = 0.001

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

    model = Autoencoder().to(device)
    model = train(
        model,
        train_loader,
        val_loader,
        learning_rate=learning_rate,
        epoch=num_epochs,
        device=device)

    model.eval()

    # def to_img(x):
    #     x = 0.5 * (x + 1)
    #     x = x.clamp(0, 1)
    #     return x

    def show_image(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualise_output(images, model):
        with torch.no_grad():
            images = images.to(device)
            images = model(images)
            images = images.cpu()
            # images = to_img(images)
            np_imagegrid = torchvision.utils.make_grid(
                images[1:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.show()

    images, labels = next(iter(val_loader))

    print('Original images')
    show_image(torchvision.utils.make_grid(images[1:50], 10, 5))
    plt.show()
    print('Autoencoder reconstruction:')
    visualise_output(images, model)
# %%
