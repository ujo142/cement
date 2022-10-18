import pytorch_lightning as pl
import torch as nn
import mlflow
from torch.utils.data import DataLoader, random_split
import torch # normalizacja albo nn albo torch
import torch.nn as nn


class CementRegressor(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.layer1 = torch.nn.Linear(4, 8)
        self.layer2 = torch.nn.Linear(8, 12)
        self.layer3 = torch.nn.Linear(12,8)
        self.layer4 = torch.nn.Linear(8,1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.05)
        self.batchNorm12 = torch.nn.BatchNorm1d(12, affine=True)
        self.batchNorm32 = torch.nn.BatchNorm1d(128, affine=True)
        self.batchNorm8 = torch.nn.BatchNorm1d(8, affine=True)
        
        
    def forward(self, x):
        x = self.layer1(x)
        #x = self.batchNorm8(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer2(x)
      #  x = self.batchNorm12(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer3(x)
        #x = self.batchNorm8(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.relu(x)
        return x


    def training_step(self, batch, batch_nb):
        x, y = batch
        #x = x.view(-1,)
        criterion = nn.MSELoss()
        logits = self(x)
       # logits = logits.view(1,)
        loss = criterion(logits,y).view(-1,1)
       # mlflow.log_metric("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        criterion = nn.MSELoss()
        logits = self(x)
        logits = logits.view(-1,)
        loss = criterion(logits,y)
        valid_rmse = torch.sqrt(criterion(logits, y))
        self.log("validation_loss", loss, logger=True, on_epoch=True)
       # mlflow.log_metric("val_loss", loss)
        self.log("validation_rmse", valid_rmse, logger=True, on_epoch=True)
      #  mlflow.log_metric("val_rmse", valid_rmse)

        mae = nn.L1Loss()
        mae_loss = mae(logits, y)
      #  mlflow.log_metric("val_mae", mae_loss)

    def test_step(self, batch, batch_nb):
        x, y = batch
        criterion = nn.MSELoss()
        logits = self(x)
        logits = logits.view(1,)
        loss = criterion(self(x),y)
        test_rmse = torch.sqrt(criterion(logits, y))
        self.log("test_loss", loss, on_epoch=True)
       # mlflow.log_param("test_loss", loss)
        self.log("test_rmse", test_rmse, on_epoch=True)
       # mlflow.log_param("test_rmse", test_rmse)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.6)
        #return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-08, weight_decay=0.6, momentum=0.95, centered=False, foreach=None)
      # lr=0.02,  lr=0.05,betas=(0.9,0.999), eps=5e-08,amsgrad=False, weight_decay=0.75


"""
    def train_dataloader(self):
        train_dl = DataLoader(self.train_data, batch_size=self.batch_size)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.test_data, batch_size=self.batch_size)
        return val_dl
"""