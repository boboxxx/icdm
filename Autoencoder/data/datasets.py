# import cv2
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
import math

NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def get_dataset(config, is_interference=False, concat_train=1, concat_test=1):
    if is_interference:
        dataset_name = config.DATA.INFDATA
        img_size = config.DATA.INF_IMG_SIZE
    else:
        dataset_name = config.DATA.DATASET
        img_size = config.DATA.IMG_SIZE
    if dataset_name == "CIFAR10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="/mnt/sda/datasets/CIFAR10/", train=True, transform=transform_train, download=False
        )

        test_dataset = datasets.CIFAR10(
            root="/mnt/sda/datasets/CIFAR10/", train=False, transform=transform_test, download=False
        )
        train_dataset = data.ConcatDataset([train_dataset] * concat_train)
        test_dataset = data.ConcatDataset([test_dataset] * concat_test)

    elif dataset_name == "DIV2K":
        transform_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/DIV2K/DIV2K_train_HR",
            transform=transform_train,
        )

        test_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/DIV2K/DIV2K_valid_HR", transform=transform_test
        )
    elif dataset_name == "OpenImg":
        transform_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/OpenImg/selected/",
            transform=transform_train,
        )

        test_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/Kodak/", transform=transform_test
        )
    elif dataset_name == "CelebA":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/CelebA/Img/trainset", transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/CelebA/Img/validset", transform=transform_test
        )

    elif dataset_name == "STL10":
        transform_train = transforms.Compose(
            [
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/stl10/train/", transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/stl10/eval/", transform=transform_test
        )
    elif dataset_name == "Bird":
        transform_train = transforms.Compose(
            [
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/Bird/train/", transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            root="/mnt/sda/datasets/Bird/test/", transform=transform_test
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,
        batch_size=config.DATA.TRAIN_BATCH,
        worker_init_fn=worker_init_fn_seed,
        shuffle=True,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=config.DATA.TEST_BATCH, shuffle=False, drop_last=True
    )

    return train_loader, test_loader


def get_loader(config, need_padding=True):

    train_loader, test_loader = get_dataset(config, is_interference=False)
    inf_train_loader, inf_test_loader = get_dataset(config, is_interference=True)
    if need_padding:
        if len(inf_train_loader) < len(train_loader):
            concat_train = math.ceil(len(train_loader) / len(inf_train_loader))
            concat_test = math.ceil(len(test_loader) / len(inf_test_loader))
            print(concat_train, concat_test)
            inf_train_loader, inf_test_loader = get_dataset(
                config, is_interference=True, concat_train=concat_train, concat_test=concat_test
            )

    return train_loader, test_loader, inf_train_loader, inf_test_loader


class config:
    dataset = "CelebA"
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    device_ids = [0]
    if_sample = False
    # logger
    print_step = 39
    plot_step = 10000
    # filename = datetime.now().__str__()[:-16]
    models = "E:\code\DDPM\SemDiffusion\Autoencoder\history"
    logger = None
    equ = "MMSE"
    # training details
    normalize = False
    learning_rate = 0.0001
    epoch = 20

    save_model_freq = 20
    if dataset == "CIFAR10":
        image_dims = (3, 32, 32)
        train_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
        test_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
    elif dataset == "DIV2K":
        image_dims = (3, 256, 256)
        train_data_dir = r"D:\dateset\DIV2K\DIV2K_train_HR"
        test_data_dir = r"D:\dateset\DIV2K\DIV2K_valid_HR"
    elif dataset == "CelebA":
        image_dims = (3, 128, 128)
        train_data_dir = r"D:\dateset\CelebA\Img\trainset"
        test_data_dir = r"D:\dateset\CelebA\Img\validset"
    batch_size = 1
    # batch_size = 100
    downsample = 4


if __name__ == "__main__":
    train_loader, test_loader = get_loader(config)
    image = next(iter(train_loader))[0]
    print(image)
