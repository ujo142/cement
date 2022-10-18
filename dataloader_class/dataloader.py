import os
from typing import Tuple, List
from torch.utils.data import DataLoader, random_split
import hydra
import torch


class CementDataloader:
    """
    DataloaderCreator creates a dataset and split it into train and val subsets.
    """

    def __init__(self, Dataset, train_data_path, val_data_path, ratio, batch_size, workers):
        self.Dataset = hydra.utils.get_class(Dataset)
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.ratio = ratio
        self.batch_size = batch_size
        self.workers = workers

    def _get_split_length(self, dataset: torch.utils.data.ConcatDataset) -> Tuple[int, int]:
        train_val_ratio = self.ratio
        train_len = round(len(dataset) * train_val_ratio)
        val_len = len(dataset) - train_len
        return train_len, val_len

    def get_dataloaders(self):
        train = self.Dataset(self.train_data_path)

        if self.val_data_path in [None, False, ""]:
            train_len, val_len = self._get_split_length(train)

            train, val = random_split(train, [train_len, val_len], generator=torch.Generator().manual_seed(0))
        else:
            val = self.Dataset(self.val_data_path)

        dataloader_train = DataLoader(
            train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            drop_last=False,
        )

        dataloader_val = DataLoader(
            val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.workers,
            drop_last=False,
        )
        return dataloader_train, dataloader_val
