import torch
from generator import Generator
from discriminator import Discriminator


def main():
    batch_size = 8
    latent_dim = 100

    G = Generator(latent_dim=latent_dim)
    D = Discriminator()

    noise = torch.randn(batch_size, latent_dim, 1, 1)

    fake_images = G(noise)
    print("Fake image shape:", fake_images.shape)

    output = D(fake_images)
    print("Discriminator output shape:", output.shape)


if __name__ == "__main__":
    main()
