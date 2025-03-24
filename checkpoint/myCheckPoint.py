import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from exp.baseExpirement import BasicExpirement
from datetime import datetime
import os
import torch
import logging
import queue
import shutil


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = logging.getLogger()
        self.__save_queue: queue.Queue = queue.Queue()
        shutil.rmtree(kwargs["dirpath"])

    def __save_model(self, trainer, pl_module, checkpoint) -> str:
        # 获取当前日期，格式化为 YYYY-MM-DD
        current_date = datetime.now().strftime("%Y_%m_%d")

        # 获取训练损失（假设已经在日志中记录了）
        recons_loss = trainer.callback_metrics.get(
            BasicExpirement.VAL_RECONS_LOSS, None
        )
        if recons_loss is not None:
            recons_loss = recons_loss.item()  # 获取标量值

        # 构建自定义文件名
        epoch = trainer.current_epoch
        filename = (
            f"model_{epoch:02d}_recons_loss_{recons_loss:.2f}_{current_date}.ckpt"
        )

        # 打印日志
        self.__logger.info(
            f"Saving checkpoint at epoch {epoch}, loss: {recons_loss:.2f}, path: {filename}"
        )

        # 更新文件路径
        checkpoint_path = os.path.join(self.dirpath, filename)
        os.makedirs(self.dirpath, exist_ok=True)
        # 保存模型

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.__save_queue.qsize() < 3:
            self.__save_queue.put(self.__save_model(trainer, pl_module, checkpoint))
        else:
            os.remove(self.__save_queue.get())
            self.__save_queue.put(self.__save_model(trainer, pl_module, checkpoint))
        return checkpoint
