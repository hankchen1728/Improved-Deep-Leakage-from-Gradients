import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, channels=1, transforms=None, mode='train'):
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
        self.channels = channels

    def __getitem__(self, index):
        label_as_tensor = self.labels[index]
        img_path = os.path.join(self.data.iloc[index]['Path'])

        img_as_np = plt.imread(img_path)
        # raw_shape = img_as_np.shape
        if self.channels != 1:
            img_as_np = np.array(
                [img_as_np] * self.channels
            ).transpose(1, 2, 0)

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
