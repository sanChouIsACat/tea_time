import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from exp.baseExpirement import BasicExpirement
from datetime import datetime
import os
import torch
import logging
import queue
import shutil
import asyncio
from typing import *
from threading import RLock, Timer


# used in high frquence save scence
class HighFrqCheckpoint(ModelCheckpoint):
    def __init__(self, save_interval=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = logging.getLogger()
        self.__to_be_saved_queue: queue.Queue = queue.Queue()
        self.__saved_queue: queue.Queue = queue.Queue()
        self.__dirpath: str = kwargs["dirpath"]
        self.__save_interval: int = save_interval
        self.__q_lock = RLock()
        self.__save_top_k = kwargs["save_top_k"]

        # init dir
        shutil.rmtree(kwargs["dirpath"], ignore_errors=True)
        os.makedirs(self.dirpath, exist_ok=True)

        # init save thread
        #Timer(self.__save_interval, self.__save_model).start()

    def __save_model(self):
        self.__logger.info("currently store mem cache")

        to_be_saved_ele = self.__to_be_saved_queue.qsize()
        for i in range(0, to_be_saved_ele):
            if self.__saved_queue.empty():
                break
            os.remove(self.__saved_queue.get())

        self.__q_lock.acquire()
        while not self.__to_be_saved_queue.empty():
            file_path, checkpoint = self.__to_be_saved_queue.get()
            torch.save(checkpoint, file_path)
            self.__saved_queue.put(file_path)
        self.__q_lock.release()

        Timer(self.__save_interval, self.__save_model).start()

    def __gen_model_path(self, trainer, pl_module, checkpoint) -> str:
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
        checkpoint_path = os.path.join(self.__dirpath, filename)
        # 保存模型

        return checkpoint_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.__q_lock.acquire()
        if self.__to_be_saved_queue.qsize() >= self.__save_top_k:
            self.__to_be_saved_queue.get()
        self.__to_be_saved_queue.put(
            (self.__gen_model_path(trainer, pl_module, checkpoint), checkpoint)
        )
        self.__q_lock.release()
        return checkpoint
