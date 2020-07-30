import os
import json

import torch
import torch.utils.data as data
import numpy as np


class patches_dataset(data.Dataset):
    def __init__(self, patches_root, gt_root):
        self.gt = np.load(gt_root)
        self.patches = np.load(patches_root)

        if self.gt.shape[0] != self.patches.shape[0]:
            print("The number of patches and gt labels does not match")
            exit(0)

    def __getitem__(self, index):
        gt = self.gt[index]
        gt = torch.from_numpy(gt).type(torch.LongTensor).squeeze()
        patch = self.patches[index]
        patch = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0)

        return patch, gt

    def __len__(self):
        return self.gt.shape[0]


def get_data_loader(patches_root, gt_root, batch_size, shuffle=True, num_workers=8):

    dataset = patches_dataset(patches_root, gt_root)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    return data_loader
