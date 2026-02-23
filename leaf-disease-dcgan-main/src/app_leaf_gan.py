import streamlit as st
import torch
import os
from torchvision.utils import save_image
from generator import Generator


CHECKPOINT = "checkpoints/G_epoch_20.pth"
LATENT_DIM = 100
OUTPUT_DIR = "generated"


def generate_images(num_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G = Generator(latent_dim=LATENT_DIM).to(device)
    G.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    G.eval()

    noise = torch.randn(num_images, LATENT_DIM, 1, 1).to(device)

    with torch.no_grad():
        fake_images = G(noise)

    output_path = os.path.join(OUTPUT_DIR, "synthetic_leaves.png")

    save_image(
        fake_images,
        output_path,
        normalize=True,
        nrow=int(num_images ** 0.5)
    )

    return output_path


st.title("🌱 Crop Leaf Disease Image Generator")
st.write("Generate synthetic diseased leaf images using DCGAN.")

num_images = st.slider("Number of images", 4, 64, 16, step=4)

if st.button("Generate Images"):
    path = generate_images(num_images)
    st.image(path, caption="Synthetic Leaf Images", use_column_width=True)
