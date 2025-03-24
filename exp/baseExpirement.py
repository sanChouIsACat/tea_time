from pytorch_lightning import LightningModule
from model.basic_model import VAE, VaeCtx
from dataclasses import dataclass
from util.serializer import serialiable
import torch
from torch import optim
import logging


@serialiable
@dataclass
class TrainCtx:
    lr: float


class BasicExpirement(LightningModule):
    TOTAL_LOSS = "total_loss"
    RECONS_LOSS = "recons_loss"
    KLD_LOSS = "kld_loss"
    VAL_RECONS_LOSS = "val_loss"

    def __init__(self, model: VAE, ctx: TrainCtx) -> None:
        super(BasicExpirement, self).__init__()
        self.__project_logger = logging.getLogger()
        self.__model = model
        self.__ctx = ctx

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.__model(input)

    def log_loss(self, total_loss, recons_loss, kld_loss, stage: str, batch_idx: str):
        self.log(BasicExpirement.TOTAL_LOSS, total_loss)
        self.log(BasicExpirement.RECONS_LOSS, recons_loss)
        self.log(BasicExpirement.KLD_LOSS, kld_loss)
        self.__project_logger.info(
            f"currently {stage} total loss [{total_loss.item()}] recons_loss [{recons_loss.item()}] kld_loss [{kld_loss.item()}] at batch [{batch_idx}]"
        )

    def training_step(self, batch, batch_idx):
        datas, labels = batch
        datas: torch.Tensor = datas

        results, mu, logvar = self.forward(datas)
        train_loss, recons_loss, kld_loss = self.__model.loss_f(
            results, labels, mu, logvar
        )

        self.log_loss(train_loss, recons_loss, kld_loss, "train", batch_idx)

        return train_loss

    def on_epoch_end(self):
        # 获取训练和验证损失的均值
        avg_total_loss = self.logged_metrics.get(
            BasicExpirement.TOTAL_LOSS, None
        ).mean()
        avg_kld_loss = self.logged_metrics.get(BasicExpirement.KLD_LOSS, None).mean()
        avg_recons_loss = self.logged_metrics.get(
            BasicExpirement.RECONS_LOSS, None
        ).mean()
        self.__project_logger.info(
            f"Epoch [{self.current_epoch + 1}] recons Loss: "
            f"[{avg_recons_loss:.4f}] kld Loss: [{avg_kld_loss:.4f} total_loss: [{avg_total_loss:.4f}]]"
        )

    def on_train_start(self):
        self.__project_logger.info(f"train started")

    def on_train_end(self):
        self.__project_logger.info(f"train has end")

    def validation_step(self, batch, batch_idx):
        datas, labels = batch
        datas: torch.Tensor = datas

        results, mu, logvar = self.forward(datas)
        val_loss, recons_loss, kld_loss = self.__model.loss_f(
            results, labels, mu, logvar
        )

        self.log_loss(val_loss, recons_loss, kld_loss, "train", batch_idx)
        self.log(BasicExpirement.VAL_RECONS_LOSS, recons_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.__model.parameters(),
            lr=self.__ctx.lr,
        )
        return optimizer
