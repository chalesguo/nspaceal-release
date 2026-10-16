def get_num_images(dataset):
    sizes = {
        "cifar10": 50000,
        "cifar10_imbalanced": 50000,
        "cifar100": 50000,
        "cifar100_imbalanced": 50000,
        "svhn": 73257,
        "caltech101": 6907,
        "caltech256": 27607,
    }
    if dataset not in sizes:
        raise NotImplementedError(f"Unknown dataset: {dataset}")
    return sizes[dataset]


cls_dict = {
    "cifar10": 10,
    "cifar10_imbalanced": 10,
    "cifar100": 100,
    "cifar100_imbalanced": 100,
    "svhn": 10,
    "caltech101": 102,
    "caltech256": 256,
}
