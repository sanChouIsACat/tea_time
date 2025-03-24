import torch.utils.data as data
import pandas as pd
import torch
from util.serializer import serialiable
from dataclasses import dataclass
from torch.utils.data import DataLoader
import pytorch_lightning


@serialiable
@dataclass
class MyLoaderCtx:
    filePath: str  # Excel文件路径
    train_percentage: float  # 训练集所占总数据集比例
    batch_size: float
    sheet_name: str = "Sheet1"  # 默认读取的sheet名称


class Mydataset(data.Dataset):
    def __init__(self, data_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.data = data_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return both data and label as a tuple
        return self.data[idx], self.labels[idx]


class MyLoader(pytorch_lightning.LightningDataModule):
    def __init__(self, loaderCtx: MyLoaderCtx):
        super().__init__()
        self.__ctx = loaderCtx

    def prepare_data(self):
        df = pd.read_excel(
            self.__ctx.filePath, sheet_name=self.__ctx.sheet_name, header=None
        )
        # Ensure the data is numeric and handle any non-numeric values (e.g., convert to NaN)
        df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

        # If there are NaN values, you can choose to either drop them or fill them with a default value
        # For example, filling NaNs with 0:
        self.__df = df.dropna(how="all")

        # Convert to tensor
        train_size = int(self.__df.shape[0] * self.__ctx.train_percentage)
        self.__train_data = torch.tensor(
            self.__df.iloc[:train_size, :-1].values, dtype=torch.float32
        )
        self.__train_data_label = torch.tensor(
            self.__df.iloc[:train_size, -1].values, dtype=torch.float32
        ).view(-1, 1)
        self.__test_data = torch.tensor(
            self.__df.iloc[train_size:, :-1].values, dtype=torch.float32
        )
        self.__test_data_label = torch.tensor(
            self.__df.iloc[train_size:, -1].values, dtype=torch.float32
        ).view(-1, 1)

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        # Use Mydataset to return both data and labels
        train_dataset = Mydataset(self.__train_data, self.__train_data_label)
        return DataLoader(train_dataset, batch_size=self.__ctx.batch_size, shuffle=True)

    def val_dataloader(self):
        # Use Mydataset to return both data and labels
        val_dataset = Mydataset(self.__test_data, self.__test_data_label)
        return DataLoader(val_dataset, batch_size=self.__ctx.batch_size, shuffle=False)
