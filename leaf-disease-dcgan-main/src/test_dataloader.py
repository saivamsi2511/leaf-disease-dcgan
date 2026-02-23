from data_loader import get_dataloader

def main():
    dataloader, classes = get_dataloader(split="train", batch_size=8)

    print("Classes:", classes)

    for images, labels in dataloader:
        print("Image batch shape:", images.shape)
        print("Labels:", labels)
        break

if __name__ == "__main__":
    main()
