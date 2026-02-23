import torch
import os
from torchvision.utils import save_image

from generator import Generator


def generate_images(
    checkpoint_path="checkpoints/G_epoch_20.pth",
    num_images=16,
    output_dir="generated",
    latent_dim=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load generator
    G = Generator(latent_dim=latent_dim).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    # Generate images
    noise = torch.randn(num_images, latent_dim, 1, 1).to(device)

    with torch.no_grad():
        fake_images = G(noise)

    # Save grid
    save_image(
        fake_images,
        os.path.join(output_dir, "synthetic_leaves.png"),
        normalize=True,
        nrow=4
    )

    print(f"{num_images} images generated and saved to {output_dir}")


if __name__ == "__main__":
    generate_images()
