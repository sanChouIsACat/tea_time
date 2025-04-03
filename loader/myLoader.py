import logging
import os
import shutil
from sympy import true
import torch.utils.data as data
import pandas as pd
import torch
from util.serializer import serialiable
from dataclasses import dataclass
from torch.utils.data import DataLoader
import pytorch_lightning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


@serialiable
@dataclass
class MyLoaderCtx:
    filePath: str  # Excel文件路径
    train_percentage: float  # 训练集所占总数据集比例
    batch_size: float
    sheet_name: str = "Sheet1"  # 默认读取的sheet名称
    split_data_dir: str = "split_datas"


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
        self.__logger = logging.getLogger()

    def prepare_data(self):
        df = pd.read_excel(
            self.__ctx.filePath, sheet_name=self.__ctx.sheet_name, header=None
        )

        # 确保数据是数值型，非数值的转换为 NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # 去除全 NaN 行
        self.__df = df.dropna(how="all")

        # 分离特征 (X) 和标签 (y)
        X = self.__df.iloc[:, :-1].values
        y = self.__df.iloc[:, -1].values

        # **Step 1: Log 变换**
        X_log_scaled = torch.log(torch.tensor(X, dtype=torch.float32) + 1)

        # **Step 2: Robust Scaling**
        scaler = RobustScaler()
        X_robust_scaled = scaler.fit_transform(X_log_scaled.cpu().numpy())

        # 重新转换为 Tensor
        X_robust_scaled = torch.tensor(X_robust_scaled, dtype=torch.float32)

        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(
            X_robust_scaled.cpu().numpy(),
            y,
            test_size=1 - self.__ctx.train_percentage,
            random_state=42,
        )

        # Save train and test datasets to CSV files
        shutil.rmtree(self.__ctx.split_data_dir, True)
        os.makedirs(self.__ctx.split_data_dir)
        train_data_df = pd.DataFrame(X_train)
        train_data_df['label'] = y_train
        train_data_df.to_csv(f"{self.__ctx.split_data_dir}/train_data.csv", index=False)

        test_data_df = pd.DataFrame(X_test)
        test_data_df['label'] = y_test
        test_data_df.to_csv(f"{self.__ctx.split_data_dir}/test_data.csv", index=False)

        # 转换为 PyTorch Tensor
        self.__train_data = torch.tensor(X_train, dtype=torch.float32)
        self.__train_data_label = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.__test_data = torch.tensor(X_test, dtype=torch.float32)
        self.__test_data_label = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        self.__logger.info("train label")
        self.__logger.info(self.__train_data_label)
        self.__logger.info("test label")
        self.__logger.info(self.__test_data_label)

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        # Use Mydataset to return both data and labels
        train_dataset = Mydataset(self.__train_data, self.__train_data_label)
        real_batch_size = min(self.__ctx.batch_size, len(train_dataset))
        return DataLoader(
            train_dataset,
            batch_size=real_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )

    def val_dataloader(self):
        # Use Mydataset to return both data and labels
        val_dataset = Mydataset(self.__test_data, self.__test_data_label)
        real_batch_size = min(self.__ctx.batch_size, len(val_dataset))
        return DataLoader(
            val_dataset,
            batch_size=real_batch_size,
            shuffle=False,
            generator=torch.Generator(device="cuda"),
        )
