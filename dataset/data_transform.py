import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .imbalanced_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


NORM = {
    "cifar10": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "cifar10_imbalanced": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "cifar100": [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    "cifar100_imbalanced": [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    "caltech101": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "caltech256": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "svhn": [[0.4376, 0.4437, 0.4728], [0.1980, 0.2010, 0.1970]],
}


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)


class ImageFolderByCategory(Dataset):
    def __init__(self, root, transform=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Caltech directory not found: {root}")
        self.root = root
        self.transform = transform
        self.categories = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.files = []
        self.y = []
        for class_id, category in enumerate(self.categories):
            category_dir = os.path.join(root, category)
            for name in sorted(os.listdir(category_dir)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(category_dir, name))
                    self.y.append(class_id)
        if not self.files:
            raise FileNotFoundError(f"No images found under: {root}")
        self.targets = self.y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        target = self.y[index]
        if self.transform:
            image = self.transform(image)
        return image, target


def split_train_test(dataset, train_ratio=0.9):
    train_indices, test_indices = [], []
    indices = range(len(dataset))
    for class_id in range(len(dataset.categories)):
        class_indices = [idx for idx in indices if dataset.y[idx] == class_id]
        np.random.shuffle(class_indices)
        split = int(np.floor(train_ratio * len(class_indices)))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])
    return train_indices, test_indices


def _caltech_root(path, dataset):
    if dataset == "caltech101":
        candidates = [
            os.path.join(path, "caltech101", "101_ObjectCategories"),
            os.path.join(path, "caltech-101", "101_ObjectCategories"),
            os.path.join(path, "caltech-101", "caltech-101", "101_ObjectCategories"),
            os.path.join(path, "101_ObjectCategories"),
        ]
    else:
        candidates = [
            os.path.join(path, "caltech256", "256_ObjectCategories"),
            os.path.join(path, "caltech256", "Caltech256"),
            os.path.join(path, "256_ObjectCategories"),
        ]
    return next((candidate for candidate in candidates if os.path.isdir(candidate)), candidates[0])


def _build_transforms(dataset, image_size=None):
    mean, std = NORM[dataset]
    if dataset.startswith("caltech"):
        size = image_size or 128
        train_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return train_transform, test_transform

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform


def get_data(path, data, image_size=None):
    train_transform, test_transform = _build_transforms(data, image_size=image_size)

    if data == "cifar10":
        train_ds = CIFAR10(path, train=True, download=True, transform=train_transform)
        eval_ds = CIFAR10(path, train=True, download=False, transform=test_transform)
        test_ds = CIFAR10(path, train=False, download=True, transform=test_transform)
    elif data == "cifar10_imbalanced":
        train_ds = IMBALANCECIFAR10(path, train=True, transform=train_transform)
        eval_ds = IMBALANCECIFAR10(path, train=True, transform=test_transform)
        test_ds = IMBALANCECIFAR10(path, train=False, transform=test_transform)
    elif data == "cifar100":
        train_ds = CIFAR100(path, train=True, download=True, transform=train_transform)
        eval_ds = CIFAR100(path, train=True, download=False, transform=test_transform)
        test_ds = CIFAR100(path, train=False, download=True, transform=test_transform)
    elif data == "cifar100_imbalanced":
        train_ds = IMBALANCECIFAR100(path, train=True, transform=train_transform)
        eval_ds = IMBALANCECIFAR100(path, train=True, transform=test_transform)
        test_ds = IMBALANCECIFAR100(path, train=False, transform=test_transform)
    elif data == "svhn":
        train_ds = SVHN(path, split="train", download=True, transform=train_transform)
        eval_ds = SVHN(path, split="train", download=True, transform=test_transform)
        test_ds = SVHN(path, split="test", download=True, transform=test_transform)
    elif data in {"caltech101", "caltech256"}:
        dataset = ImageFolderByCategory(_caltech_root(path, data))
        train_idx, test_idx = split_train_test(dataset)
        train_ds = SubsetWithTransform(Subset(dataset, train_idx), transform=train_transform)
        eval_ds = SubsetWithTransform(Subset(dataset, train_idx), transform=test_transform)
        test_ds = SubsetWithTransform(Subset(dataset, test_idx), transform=test_transform)
    else:
        raise NotImplementedError(f"Dataset is not supported in this release: {data}")

    print(f"{data} | Train={len(train_ds)} Test={len(test_ds)}")
    return train_ds, test_ds, eval_ds
