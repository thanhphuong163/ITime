from typing import Optional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from datasets import MyDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        window_size: int,
        batch_sz: int
        ):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.bacth_sz = batch_sz

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full_dataset = MyDataset(self.data_dir, stage, self.window_size)
            data_length = self.full_dataset.__len__()
            train_size = int(0.7*data_length)
            val_size = data_length - train_size
            self.train_set, self.val_set = random_split(
                self.full_dataset,
                lengths=[train_size, val_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = MyDataset(self.data_dir, stage, self.window_size)

        if stage == "predict" or stage is None:
            self.predict_set = MyDataset(self.data_dir, stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bacth_sz)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bacth_sz)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bacth_sz)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.bacth_sz)