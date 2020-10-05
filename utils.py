import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def load_image(file):
    img = cv2.imread(str(file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def t2d(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed: int = 42) -> None:
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = True


class ImagesDataset(Dataset):
    def __init__(self, images, targets=None, transforms=None):
        self.images = images
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file = str(self.images[idx])
        img = load_image(file)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        if self.targets is None:
            return img

        target = self.targets[idx]
        return img, target
