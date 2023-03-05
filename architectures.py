# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent_dims=2):
        super(Autoencoder, self).__init__()
        self.c = 4
        self.latent_dims = latent_dims

        # Encoder block
        # output_height = ([input_height - kernel_size + (2*padding)] / stride) + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(in_features=self.c*7*7, out_features=latent_dims)  # c*2*7*7

        # Decoder block
        self.fc_dec = nn.Linear(in_features=latent_dims, out_features=self.c * 7 * 7)

        # output_height = [(input_height - 1) * stride] + kernel_size[0] - [2 * padding] + output_padding
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

        x = self.fc_dec(encoded)
        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.c, 7, 7)
        x4 = F.relu(self.trans_conv5(x))
        x5 = F.relu(self.trans_conv6(x4))
        x6 = F.relu(self.trans_conv7(x5))

        decoded = (self.trans_conv8(x6))
        # decoded = F.relu(self.trans_conv8(x6))
        return decoded


class Autoencoder2(nn.Module):
    def __init__(self, latent_dims=2):
        super(Autoencoder2, self).__init__()
        self.c = 4
        self.latent_dims = latent_dims

        # Encoder block
        # input: (b, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1) 
        # (b, c, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        # (b, c, 7, 7)

        # self.conv3 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
        # # (b, c, 7, 7)

        # self.conv4 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
        # # (b, c, 7, 7)

        # self.fc = nn.Linear(in_features=self.c*7*7,
        #                     out_features=latent_dims)  # c*2*7*7

        # # Decoder block
        # self.fc_dec = nn.Linear(in_features=latent_dims,
        #                         out_features=self.c * 7 * 7)

        # self.trans_conv5 = nn.ConvTranspose2d(
        #     in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
        # self.trans_conv6 = nn.ConvTranspose2d(
        #     in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)

        # output_height = [(input_height - 1) * stride] + kernel_size[0] - [2 * padding] + output_padding
        self.trans_conv7 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1, output_padding=1)
        # output size = [(7 - 1) * 2] + 3 - [2 * 1] + 1 = 14
        
        self.trans_conv8 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # output size = [(14 - 1) * 2] + 3 - [2 * 1] + 1 = 28

    def forward(self, x):  # (b, 1, 28, 28)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        # x3 = F.relu(self.conv3(x2))
        # encoded = F.relu(self.conv4(x3))  # (b, 1, 7, 7)

        # # flatten batch of multi-channel feature maps to a batch of feature vectors
        # encoded = encoded.view(encoded.size(0), -1)
        # encoded = self.fc(encoded)

        # x = self.fc_dec(encoded)
        # # unflatten batch of feature vectors to a batch of multi-channel feature maps
        # x = x.view(x.size(0), self.c, 7, 7)
        # x4 = F.relu(self.trans_conv5(x))
        # x5 = F.relu(self.trans_conv6(x4))

        x6 = F.relu(self.trans_conv7(x2))
        decoded = (self.trans_conv8(x6))
        return decoded


# x = torch.randn(1, 1, 28, 28)
# model = Autoencoder()
# print(model(x).shape)


# %%

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
