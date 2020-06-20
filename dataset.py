import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib
from torchvision import datasets, transforms

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mode='train'):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """

        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data['Support Devices'])
        self.batch_id = 0

        self.transforms = transforms

    def __getitem__(self, index):
        label_as_tensor = self.labels[index]
        img_path = os.path.join(self.data.iloc[index]['Path'])

        img_as_np = plt.imread(img_path)
        # raw_shape = img_as_np.shape
        img_as_np = np.array([img_as_np] * 3).transpose(1, 2, 0)

        img_as_np = Image.fromarray(img_as_np)

        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_np)
        else:
            img_as_tensor = img_as_np
        # Return image and the label

        # return (img_as_tensor.unsqueeze(2).repeat(1, 1, 3), label_as_tensor)
        return (img_as_tensor, label_as_tensor)

    def __next__(self):
        X, y = self.__getitem__(self.batch_id)

        self.batch_id += 1
        if self.batch_id >= self.__len__():
            self.batch_id = 0

        return X, y

    def __len__(self):
        return len(self.data.index)


class ImageDataset(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = ImageDataset(
            images_all, np.asarray(labels_all, dtype=int),
            transform=transform)
    return dst


class DatasetParams():
    def __init__(self, name, data_path, root_path):
        selected_indices = []
        cmap = "viridis"
        if name == 'MNIST':
            cmap = "gray"
            shape_img = (28, 28)
            num_classes = 10
            channel = 1
            # hidden = 588
            dst = datasets.MNIST(data_path, download=True)
            selected_indices = np.array([
                1,
                21,
                34,  # label : 0
                3,
                6,
                8,  # label : 1
                5,
                16,
                25,  # label : 2
                7,
                10,
                12,  # label : 3
                2,
                9,
                20,  # label : 4
                0,
                11,
                35,  # label : 5
                13,
                18,
                32,  # label : 6
                15,
                29,
                38,  # label : 7
                17,
                31,
                41,  # label : 8
                4,
                19,
                22  # label : 9
            ])
        elif name == "cifar10":
            shape_img = (32, 32)
            num_classes = 100
            channel = 3
            # hidden = 768
            dst = datasets.CIFAR10(data_path, download=True)
            # selected_indices = np.array([
            #     29, 30, 35,
            #     4,  5, 32,
            #     6, 13, 18,
            #     9, 17, 21,
            #     3, 10, 20,
            #     27, 40, 51,
            #     0, 19, 22,
            #     7, 11, 12,
            #     8, 62, 69,
            #     1,  2, 14
            # ])
            selected_indices = np.array([
                29, 30, 35, 49, 77, 93, 115, 116, 129, 165, 4, 5, 32, 44, 45,
                46, 60, 61, 64, 65, 6, 13, 18, 24, 41, 42, 47, 48, 54, 55, 9,
                17, 21, 26, 33, 36, 38, 39, 59, 74, 3, 10, 20, 28, 34, 58, 66,
                82, 86, 89, 27, 40, 51, 56, 70, 81, 83, 107, 128, 148, 0, 19,
                22, 23, 25, 72, 95, 103, 104, 117, 7, 11, 12, 37, 43, 52, 68,
                73, 84, 85, 8, 62, 69, 92, 100, 106, 111, 135, 139, 155, 1, 2,
                14, 15, 16, 31, 50, 53, 67, 71
            ])
            num_exp = selected_indices.shape[0]
        elif name == 'cifar100':
            shape_img = (32, 32)
            num_classes = 100
            channel = 3
            # hidden = 768
            dst = datasets.CIFAR100(data_path, download=True)

        elif name == 'lfw':
            shape_img = (32, 32)
            num_classes = 5749
            channel = 3
            lfw_path = os.path.join(root_path, '../data/lfw')
            dst = lfw_dataset(lfw_path, shape_img)

        elif name == 'CheXpert':
            shape_img = (112, 112)
            num_classes = 2
            channel = 3
            resize_t = transforms.Resize(shape_img)
            dst = CheXpertDataset(csv_path='./idlg_data_entry.csv',
                                  transforms=resize_t)
            selected_indices = np.arange(0, 30, 3)
        else:
            exit('unknown dataset')
