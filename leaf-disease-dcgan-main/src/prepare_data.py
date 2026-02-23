import os
import shutil
import random
import yaml

CONFIG_PATH = "configs/data_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def split_data():
    config = load_config()

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    classes = config["classes"]

    train_ratio = config["split"]["train"]
    val_ratio = config["split"]["val"]

    for cls in classes:
        class_path = os.path.join(raw_dir, cls)
        images = os.listdir(class_path)

        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_images in splits.items():
            split_class_dir = os.path.join(
                processed_dir, split_name, cls
            )
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy(src, dst)

        print(f"Finished splitting class: {cls}")


if __name__ == "__main__":
    split_data()
