import os
from re import A
import mlflow
import mlflow.pytorch
import torchvision
from dataset import CemDataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
import pytorch_lightning as pl

from model import CementPredictor
#from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST

def main():
      train_data = CemDataset("./datasets/train_scaled.csv")
      test_data = CemDataset("./datasets/test_scaled.csv")
      
      cement_model = CementPredictor(learning_rate=1e-3,
                                    batch_size=4,
                                    train_data=train_data,
                                    test_data=test_data)

      trainer = pl.Trainer(max_epochs=15,
                           accelerator='cpu',
                           devices=1,
                           log_every_n_steps=5,
                           enable_progress_bar=True)
      
      trainer.fit(cement_model)
      
      test_sample = torch.tensor([-0.60207995, -0.49720917,  0.66516602, -1.00230708])
      cement_model = cement_model.eval()
      testowy_wynik = cement_model(test_sample)
      print(f"Testowy_wynik: {testowy_wynik}")

if __name__ == "__main__":
    main()