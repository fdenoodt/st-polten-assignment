# %%
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from main_clean import Autoencoder, visualise_output, show_image
import numpy as np
from architectures import Autoencoder, Encoder, Decoder

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

    decoder = Decoder(latent_dim).to(device)
    decoder.load_state_dict(torch.load(f'model_latent_dims_{latent_dim}.pt'))
    decoder.eval()

    encs = torch.randn(1, latent_dim).to(device)
    encs[0][0] = -0.9
    encs[0][0] = 1

    imgs = decoder(encs)

    show_image(imgs[0], 'Generated Image', save=False)

# %%
