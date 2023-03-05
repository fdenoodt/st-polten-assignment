# %%
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from vae_experiment import VanillaVAE
from main_clean import visualise_output, show_image
import numpy as np
from architectures import Encoder, Decoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    logs_path = "temp/"

    batch_size = 128 * 2 * 2 * 2  # 1024
    mnist_val = dataset.MNIST(
        "./", train=False,
        transform=transforms.ToTensor(),
        download=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=mnist_val,
        batch_size=batch_size,
        shuffle=False)

    LATENT_DIM = 10
    MODEL: VanillaVAE = VanillaVAE(
        latent_dim=LATENT_DIM, in_channels=1).to(device)
    MODEL_PATH = f"temp/vae_model_latent_dims_{LATENT_DIM}.pt"

    # Load the saved model
    MODEL.load_state_dict(torch.load(MODEL_PATH))
    MODEL.eval()

    images, labels = next(iter(val_loader))
    images = images.to(device)
    output = MODEL(images)

    MODEL.eval()
    # show_image(torchvision.utils.make_grid(
    #     images[1:50], 10, 5), f"{logs_path}/LATENT_DIM_{LATENT_DIM}.png", save=False)

    visualise_output(images, MODEL, device,
                     f"{logs_path}/img_LATENT_DIM_{LATENT_DIM}.png", save=False)

    encoder = MODEL.decoder_cnn
    mu, log_var = MODEL.encode(images)
    hidden_representations = MODEL.reparameterize(mu, log_var)
    print(f"Hidden representations shape: {hidden_representations.shape}")

    # Create a list of 2D hidden representation
    hidden_representations = hidden_representations.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy().tolist()

    tsne = TSNE(init='random', n_components=2, random_state=0,
                learning_rate=200)  # enter init and lr to avoid warnings
    hidden_representations_2d = tsne.fit_transform(hidden_representations)

    tsne = TSNE(init='random', n_components=2,
                random_state=0, learning_rate=200)
    hidden_representations_2d = tsne.fit_transform(hidden_representations)

    colors = ['red', 'blue', 'green', 'yellow', 'purple',
              'pink', 'cyan', 'black', 'magenta', 'orange']
    
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(hidden_representations_2d[mask, 0], hidden_representations_2d[mask, 1], c=colors[int(
            label)], label=str(label))

    # Add a legend
    ax.legend()
    plt.show()