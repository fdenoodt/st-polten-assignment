# %%
# https://github.com/williamFalcon/pytorch-lightning-vae/blob/1608c979f339915de0e7149661e87c8a9e5ac9f2/vae.py#L88
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 512)
        self.linear2 = nn.Linear(512, 784)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return x

        # mu =  self.linear2(x)
        # sigma = torch.exp(self.linear3(x))
        # z = mu + sigma*self.N.sample(mu.shape)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # return z
    
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.latent_dims = latent_dims
#         self.encoder = VariationalEncoder(latent_dims)
#         self.decoder = Decoder(latent_dims)
    
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)
    

# x = torch.randn(1, 1, 28, 28).cuda()
# model = VariationalAutoencoder(2).cuda()
# print(model(x).shape)

# encoder = model.encoder
# print(encoder(x).shape)




class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, enc_out_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        # encoder, decoder
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim)
    

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.kl = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # print(x_encoded.shape, mu.shape, log_var.shape) # torch.Size([1024, 512]) torch.Size([1024, 2]) torch.Size([1024, 2])

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        # decoded
        x_hat = self.decoder(z)

        return x_hat, z, mu, std

    # def training_step(self, batch, batch_idx):
    #     x, _ = batch

    #     # encode x to get the mu and variance parameters
    #     x_encoded = self.encoder(x)
    #     mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

    #     # sample z from q
    #     std = torch.exp(log_var / 2)
    #     q = torch.distributions.Normal(mu, std)
    #     z = q.rsample()

    #     # decoded
    #     x_hat = self.decoder(z)

    #     # reconstruction loss
    #     recon_loss = self.gaussia6n_likelihood(x_hat, self.log_scale, x)

    #     # kl
    #     kl = self.kl_divergence(z, mu, std)

    #     # elbo
    #     elbo = (kl - recon_loss)
    #     elbo = elbo.mean()

    #     self.log_dict({
    #         'elbo': elbo,
    #         'kl': kl.mean(),
    #         'recon_loss': recon_loss.mean(),
    #         'reconstruction': recon_loss.mean(),
    #         'kl': kl.mean(),
    #     })

    #     return elbo


# def train():
#     parser = ArgumentParser()
#     parser.add_argument('--gpus', type=int, default=None)
#     parser.add_argument('--dataset', type=str, default='cifar10')
#     args = parser.parse_args()

#     if args.dataset == 'cifar10':
#         dataset = CIFAR10DataModule('.')
#     if args.dataset == 'imagenet':
#         dataset = ImagenetDataModule('.')

#     sampler = ImageSampler()

#     vae = VAE()
#     trainer = pl.Trainer(gpus=args.gpus, max_epochs=20, callbacks=[sampler])
#     trainer.fit(vae, dataset)


# if __name__ == '__main__':
#     train()





