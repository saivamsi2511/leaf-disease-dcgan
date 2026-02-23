import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import csv

from data_loader import get_dataloader
from generator import Generator
from discriminator import Discriminator


def train():
    # Hyperparameters
    batch_size = 64
    latent_dim = 100
    lr = 0.0002
    num_epochs = 20   # increased for better results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataloader, _ = get_dataloader(split="train", batch_size=batch_size)

    # Models
    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator().to(device)

    # Loss
    criterion = nn.BCELoss()

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Create folders
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # CSV logging
    log_file = open("training_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "loss_D", "loss_G"])

    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)

            # ------------------
            # Train Discriminator
            # ------------------
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake = G(noise)

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D_real = criterion(D_real, real_labels)
            loss_D_fake = criterion(D_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake

            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ------------------
            # Train Generator
            # ------------------
            output = D(fake)
            loss_G = criterion(output, real_labels)

            G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
        )

        # Log to CSV
        log_writer.writerow([epoch + 1, loss_D.item(), loss_G.item()])

        # Save sample images
        with torch.no_grad():
            fake_samples = G(fixed_noise)
            save_image(
                fake_samples,
                f"samples/epoch_{epoch+1}.png",
                normalize=True,
                nrow=8
            )

        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch+1}.pth")

    log_file.close()


if __name__ == "__main__":
    train()
