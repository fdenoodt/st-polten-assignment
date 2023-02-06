# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Autoencoder(nn.Module):
#     def __init__(self, latent_dims=2):
#         super(Autoencoder, self).__init__()
#         self.c = 4
#         self.latent_dims = latent_dims

#         # Encoder block
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)

#         self.fc = nn.Linear(in_features=self.c*7*7,
#                             out_features=latent_dims)  # c*2*7*7

#         # Decoder block
#         self.fc_dec = nn.Linear(in_features=latent_dims,
#                                 out_features=self.c * 7 * 7)

#         self.trans_conv5 = nn.ConvTranspose2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#         self.trans_conv6 = nn.ConvTranspose2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#         self.trans_conv7 = nn.ConvTranspose2d(
#             in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#         self.trans_conv8 = nn.Conv2d(
#             in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

#     def forward(self, x):  # (b, 1, 28, 28)
#         x1 = F.relu(self.conv1(x))
#         x2 = F.relu(self.conv2(x1))
#         x3 = F.relu(self.conv3(x2))
#         encoded = F.relu(self.conv4(x3))  # (b, 1, 7, 7)

#         # flatten batch of multi-channel feature maps to a batch of feature vectors
#         encoded = encoded.view(encoded.size(0), -1)
#         encoded = self.fc(encoded)

#         x = self.fc_dec(encoded)
#         # unflatten batch of feature vectors to a batch of multi-channel feature maps
#         x = x.view(x.size(0), self.c, 7, 7)
#         x4 = F.relu(self.trans_conv5(x))
#         x5 = F.relu(self.trans_conv6(x4))
#         x6 = F.relu(self.trans_conv7(x5))

#         decoded = (self.trans_conv8(x6))
#         # decoded = F.relu(self.trans_conv8(x6))
#         return decoded


# x = torch.randn(1, 1, 28, 28)
# model = Autoencoder()
# print(model(x).shape)
















# class Encoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Encoder, self).__init__()
#         self.linear1 = nn.Linear(784, 512)
#         self.linear2 = nn.Linear(512, latent_dims)
    
#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    
# class Autoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder(latent_dims)
#         self.decoder = Decoder(latent_dims)
    
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)





class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# %%

# class Encoder(nn.Module):
#   def __init__(self, latent_dims=2):
#       super(Encoder, self).__init__()
#       self.c = 4
#       self.latent_dims = latent_dims

#       # Encoder block
#       self.conv1 = nn.Conv2d(
#           in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.conv2 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.conv3 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
#       self.conv4 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)

#       self.fc = nn.Linear(in_features=self.c*7*7,
#                           out_features=latent_dims)  # c*2*7*7

#       # Decoder block
#       self.fc_dec = nn.Linear(in_features=latent_dims, out_features=self.c * 7 * 7)

#       self.trans_conv5 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv6 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv7 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv8 = nn.Conv2d(in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

#   def forward(self, x):  # (b, 1, 28, 28)
#       x1 = F.relu(self.conv1(x))
#       x2 = F.relu(self.conv2(x1))
#       x3 = F.relu(self.conv3(x2))
#       encoded = F.relu(self.conv4(x3))  # (b, 1, 7, 7)

#       # flatten batch of multi-channel feature maps to a batch of feature vectors
#       encoded = encoded.view(encoded.size(0), -1)
#       encoded = self.fc(encoded)
#       return encoded



# class Decoder(nn.Module):
#   def __init__(self, latent_dims=2):
#       super(Decoder, self).__init__()
#       self.c = 4
#       self.latent_dims = latent_dims

#       # Encoder block
#       self.conv1 = nn.Conv2d(
#           in_channels=1, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.conv2 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.conv3 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)
#       self.conv4 = nn.Conv2d(
#           in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1)

#       self.fc = nn.Linear(in_features=self.c*7*7,
#                           out_features=latent_dims)  # c*2*7*7

#       # Decoder block
#       self.fc_dec = nn.Linear(in_features=latent_dims, out_features=self.c * 7 * 7)

#       self.trans_conv5 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv6 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv7 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=2, padding=1)
#       self.trans_conv8 = nn.Conv2d(in_channels=self.c, out_channels=1, kernel_size=3, stride=2, padding=4)

#   def forward(self, encoding):  # (b, 1, 28, 28)
#       x = self.fc_dec(encoding)
#       # unflatten batch of feature vectors to a batch of multi-channel feature maps
#       x = x.view(x.size(0), self.c, 7, 7)
#       x4 = F.relu(self.trans_conv5(x))
#       x5 = F.relu(self.trans_conv6(x4))
#       x6 = F.relu(self.trans_conv7(x5))

#       decoded = F.relu(self.trans_conv8(x6))
#       return decoded
