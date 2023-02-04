# %%
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from main_clean import Autoencoder, visualise_output, show_image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %%

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_size = 128
    mnist_val = dataset.MNIST(
        "./", train=False,
        transform=transforms.ToTensor(),
        download=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=mnist_val,
        batch_size=batch_size,
        shuffle=False)

    # Define your model
    latent_dim = 2
    model = Autoencoder(latent_dim).to(device)

    # Load the saved model
    model.load_state_dict(torch.load(f'model_latent_dims_{latent_dim}.pt'))
    model.eval()

    images, labels = next(iter(val_loader))
    images = images.to(device)
    output = model(images)

    model.eval()
    show_image(torchvision.utils.make_grid(
        images[1:50], 10, 5), f"latent_dim_{latent_dim}.png", save=False)

    visualise_output(images, model, device, f"img_latent_dim_{latent_dim}.png", save=False)

    # %%
    # class Encoder(nn.Module):
    #     def __init__(self, model):
    #         super(Encoder, self).__init__()
    #         self.encoder = nn.Sequential(*list(model.children()))[:5] # take first 5 layers

    #     def forward(self, x):
    #         x = self.encoder(x)
    #         return x

    class Encoder(nn.Module):
      def __init__(self, latent_dims=2):
          super(Encoder, self).__init__()
          self.c = 4
          self.latent_dims = latent_dims

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
          self.fc_dec = nn.Linear(in_features=latent_dims, out_features=self.c * 7 * 7)

          self.trans_conv5 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv6 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv7 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv8 = nn.Conv2d(in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

      def forward(self, x):  # (b, 1, 28, 28)
          x1 = F.relu(self.conv1(x))
          x2 = F.relu(self.conv2(x1))
          x3 = F.relu(self.conv3(x2))
          encoded = F.relu(self.conv4(x3))  # (b, 1, 7, 7)

          # flatten batch of multi-channel feature maps to a batch of feature vectors
          encoded = encoded.view(encoded.size(0), -1)
          encoded = self.fc(encoded)
          return encoded


    encoder = Encoder(latent_dim).to(device)
    encoder.load_state_dict(torch.load(f'model_latent_dims_{latent_dim}.pt'))
    encoder.eval()
    # Obtain the hidden representation
    hidden_representations = encoder(images)
    print(hidden_representations.shape)
    print(hidden_representations)

    # %%
    # %%
    import matplotlib.pyplot as plt

    # Create a list of 2D hidden representation
    hidden_representations = hidden_representations.cpu().detach().numpy().tolist()
    labels = labels.cpu().detach().numpy().tolist()
    
    # %%
    # Plot the points using scatter plot
    plt.scatter(*zip(*hidden_representations))

    # Label the axes
    plt.xlabel('Dim1 Axis')
    plt.ylabel('Dim2 Axis')

    # Show the plot
    plt.show()

    # %%
    import matplotlib.pyplot as plt

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'cyan', 'black', 'magenta', 'orange']

    fig, ax = plt.subplots()
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(*zip(*[hidden_representations[i] for i in range(len(hidden_representations)) if mask[i]]), c=colors[int(label)], label=str(label))

    ax.legend()
    plt.show()


    # %%

    class Decoder(nn.Module):
      def __init__(self, latent_dims=2):
          super(Decoder, self).__init__()
          self.c = 4
          self.latent_dims = latent_dims

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
          self.fc_dec = nn.Linear(in_features=latent_dims, out_features=self.c * 7 * 7)

          self.trans_conv5 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv6 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv7 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
          self.trans_conv8 = nn.Conv2d(in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

      def forward(self, encoding):  # (b, 1, 28, 28)
          x = self.fc_dec(encoding)
          # unflatten batch of feature vectors to a batch of multi-channel feature maps
          x = x.view(x.size(0), self.c, 7, 7)
          x4 = F.relu(self.trans_conv5(x))
          x5 = F.relu(self.trans_conv6(x4))
          x6 = F.relu(self.trans_conv7(x5))

          decoded = F.relu(self.trans_conv8(x6))
          return decoded

    decoder = Decoder(latent_dim).to(device)
    decoder.load_state_dict(torch.load(f'model_latent_dims_{latent_dim}.pt'))
    decoder.eval()

    encs = torch.randn(1, latent_dim).to(device)
    encs[0][0] = -100
    encs[0][0] = 0.7

    imgs = decoder(encs)

    show_image(imgs[0], 'Generated Image', save=False)
