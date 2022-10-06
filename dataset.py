from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import torch
import pandas as pd

class CemDataset(Dataset):
    def __init__(self,data_path):
        self.data=pd.read_csv(data_path)
        # In case "Unnamed: 0" exists
        self.data = self.data.drop(columns=["Unnamed: 0"])
        self.x=self.data.iloc[:,:4].values
        self.y=self.data.iloc[:,4].values
        
        self.x=torch.tensor(self.x,dtype=torch.float32)
        self.y=torch.tensor(self.y,dtype=torch.float32)

        #self.x=torch.nn.functional.normalize(self.x, p=2.0, dim = 1) # CHECK

     #   self.mean = torch.mean(self.x, dim=0)
     #   self.var = torch.var(self.x, dim=0)
     #   self.x = (self.x-self.mean)/torch.sqrt(self.var)
  
    
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]


"""
  self.samples = list()
        with open(data_path, "r", encoding="utf-8") as f:
            data = csv.reader(f)
            header = next(data)
            for line in tqdm(data):
                self.samples.append(line)

    


 sample = {}
        sample["wc"] = torch.tensor(float(self.samples[idx][0]))
        sample["z28"] = torch.tensor(float(self.samples[idx][1]))
        sample["z816"] = torch.tensor(float(self.samples[idx][2]))
        sample["p02"] = torch.tensor(float(self.samples[idx][3]))
        sample["target"] = torch.tensor(float(self.samples[idx][4]))
 """  