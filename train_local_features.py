import os
from pathlib import Path
import albumentations as albu
import albumentations.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import t2d, seed_all, ImagesDataset
from models import EfficientNetEncoder, EncoderLocalFeatures


torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "imagewoof2-320"


def main():
    if not torch.cuda.is_available():
        raise ValueError("No CUDA available")

    device = torch.device("cuda:0")
    has_multiple_devices = torch.cuda.device_count() > 1

    train_batch_size = 128
    valid_batch_size = 128
    n_epochs = 35

    # loaders
    train_transforms = albu.Compose(
        [
            albu.Resize(256, 256),
            albu.RandomCrop(224, 224),
            albu.HorizontalFlip(),
            albu.Normalize(),
            albu.pytorch.ToTensorV2(),
        ]
    )

    valid_transforms = albu.Compose(
        [albu.Resize(224, 224), albu.Normalize(), albu.pytorch.ToTensorV2()]
    )

    classes = sorted(set(os.listdir(os.path.join(DATA_DIR, "train"))))
    if set(classes) != set(os.listdir(os.path.join(DATA_DIR, "val"))):
        raise ValueError("Different classes in train & valid folders!")

    class2index = {item: num for num, item in enumerate(classes)}

    def _images_and_classes(folder):
        files, targets = [], []
        for file in folder.glob("*/*.JPEG"):
            image_folder = str(file).split("/")[-2]
            files.append(str(file))
            targets.append(class2index[image_folder])
        return files, targets

    train_dataset = ImagesDataset(
        *_images_and_classes(Path(DATA_DIR) / "train"), transforms=train_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )

    valid_dataset = ImagesDataset(
        *_images_and_classes(Path(DATA_DIR) / "val"), transforms=valid_transforms
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        drop_last=False,
    )

    encoder = EfficientNetEncoder("efficientnet-b1")
    encoder.load_state_dict(
        torch.load("encoder.pth", map_location="cpu")["encoder_state_dict"]
    )

    model = EncoderLocalFeatures(encoder, 10)
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(device)
    if has_multiple_devices:
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    reconstruction_criterion = nn.MSELoss()
    attention_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        train_attention_loss = 0.0
        train_reconstruction_loss = 0.0
        for idx, batch in enumerate(train_loader):
            x, y = t2d(batch, device)
            # with autograd.detect_anomaly():
            emb, rec, cls, _ = model(x)

            attention_loss = attention_criterion(cls, y)
            train_attention_loss += attention_loss.item()

            reconstruction_loss = reconstruction_criterion(emb, rec)
            train_reconstruction_loss += reconstruction_loss.item()

            loss = attention_loss + reconstruction_loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= idx + 1
        train_attention_loss /= idx + 1
        train_reconstruction_loss /= idx + 1

        model.eval()
        valid_loss = 0.0
        valid_attention_loss = 0.0
        valid_reconstruction_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                x, y = t2d(batch, device)
                emb, rec, cls, _ = model(x)

                attention_loss = attention_criterion(cls, y)
                valid_attention_loss += attention_loss.item()

                reconstruction_loss = reconstruction_criterion(emb, rec)
                valid_reconstruction_loss += reconstruction_loss.item()

                loss = attention_loss + reconstruction_loss
                valid_loss += loss.item()

                acc = (torch.argmax(cls, -1) == y).sum().detach().item()
                acc /= y.size(0)
                accuracy += acc
        valid_loss /= idx + 1
        valid_attention_loss /= idx + 1
        valid_reconstruction_loss /= idx + 1
        accuracy /= idx + 1

        print(f"Epoch {epoch}/{n_epochs}")
        print(
            "train - {:.5f} (attention - {:.5f}, reconstruction - {:.5f})".format(
                train_loss, train_attention_loss, train_reconstruction_loss
            )
        )
        print(
            "valid - {:.5f} (attention - {:.5f}, reconstruction - {:.5f}), accuracy - {:.5f}".format(
                valid_loss, valid_attention_loss, valid_reconstruction_loss, accuracy
            )
        )

    torch.save({"model_state_dict": model.module.state_dict()}, "local_features.pth")


if __name__ == "__main__":
    main()
