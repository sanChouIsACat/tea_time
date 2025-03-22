import torch.utils.data as data
import pandas as pd
import torch
from util.serializer import serialiable
from dataclasses import dataclass


@serialiable
@dataclass
class MyLoaderCtx:
    filePath: str  # Excel文件路径
    sheet_name: str = "Sheet1"  # 默认读取的sheet名称


class MyLoader(data.Dataset):
    def __init__(self, loaderCtx: MyLoaderCtx):
        self.ctx = loaderCtx
        df = pd.read_excel(
            self.ctx.filePath, sheet_name=self.ctx.sheet_name, header=None
        )
        self.data = torch.tensor(
            df.iloc[1:, 1:].values, dtype=torch.float32
        )  # 读取数据并转换为Tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 返回Tensor
