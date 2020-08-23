import os
import numpy as np

import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

## 트랜스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = 0.5
        self.std = 0.5

    def __call__(self, data):
        input, label = data['input'], data['label']

        input = (input - self.mean) / self.std

        data = {'input': input, 'label': label}

        return data

class RandomFlip(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)

        data = {'input': input, 'label': label}

        return data