import os
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CONFIG_PATH = "configs/data_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_dataloader(split="train", batch_size=64):
    config = load_config()

    processed_dir = config["data"]["processed_dir"]
    image_size = config["image"]["size"]

    data_path = os.path.join(processed_dir, split)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )

    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

    return dataloader, dataset.classes
