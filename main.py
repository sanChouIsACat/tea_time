from util.serializer import Serializer, serialiable
from dataclasses import dataclass
from loader.myLoader import MyLoaderCtx, MyLoader
from model.basic_model import VaeCtx, VAE
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from checkpoint.myCheckPoint import MyModelCheckpoint
from exp.baseExpirement import BasicExpirement, TrainCtx
from pytorch_lightning.callbacks import EarlyStopping
import traceback
import logging
import logging.config
import os
import json


@serialiable
@dataclass
class TrainConfig:
    model_paras_log_dir: str
    project_log_dir: str
    model_save_dir: str
    vae_config: VaeCtx
    loader_config: MyLoaderCtx
    train_config: TrainCtx


def config_project_log(config_path: str, log_dir: str) -> None:

    # 读取并加载配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        log_config = json.load(f)

    # 动态设置日志文件路径
    log_config["handlers"]["rotating_file"]["filename"] = os.path.join(
        log_dir, log_config["handlers"]["rotating_file"]["filename"]
    )

    os.makedirs(log_dir, exist_ok=True)

    # 加载日志配置
    logging.config.dictConfig(log_config)


# define paras
def parse_args():
    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument(
        "--config",
        "-c",
        help="path to the config file",
        default="configs/vae.yaml",
        type=str,
    )
    parser.add_argument(
        "--log_config",
        help="path to the config file",
        default="configs/log_config.json",
        type=str,
    )
    parser.add_argument(
        "--generate_config",
        action="store_true",  # 当该选项被传递时，'generate-config' 被设为True
        help="Generate a configuration file example",
    )
    parser.add_argument(
        "--stored_model", type=str, help="path of trained model", default=None
    )
    return parser.parse_args()


def load_config(config_path: str):
    # load config
    config: TrainConfig = None
    try:
        with open(args.config, "r") as file:
            config = Serializer.deserialize(file.read(), TrainConfig)
            return config
    except Exception as exc:
        print(exc)
        print("exit now!")
        exit(1)


if __name__ == "__main__":
    args = parse_args()
    if args.generate_config:
        print("********* example ***********")
        print(Serializer.generate_example_yaml(TrainConfig))
        print("********* example ***********")
        exit(0)

    config = load_config(args.config)
    config_project_log(args.log_config, config.project_log_dir)
    print("project log system initd succeed")

    # main
    try:
        # init logger
        tb_logger = TensorBoardLogger(save_dir=config.model_paras_log_dir)
        logger = logging.getLogger()
        logger.info("log system started success")

        # init model
        logger.info("init model")
        vae_model = VAE(config.vae_config)
        logger.info("init model succeed")

        # init expirement
        logger.info("init exp")
        exp = BasicExpirement(vae_model, config.train_config)
        logger.info("init exp succeed")
        if args.stored_model != None:
            model_path: str = args.stored_model
            logger.info(f"load exp from {model_path}")
            exp.load_from_checkpoint(model_path)
            logger.info(f"load exp from checkpoint succeed")

        # load data
        logger.info("load data")
        loader = MyLoader(config.loader_config)
        logger.info("load data succeed")

        # init trainer
        logger.info("init trainer")
        runner = Trainer(
            logger=tb_logger,
            max_epochs=10000,
            callbacks=[
                EarlyStopping(
                    monitor=BasicExpirement.VAL_RECONS_LOSS,  # 监控验证损失
                    patience=100,  # 如果3个epoch内损失没有改善，则提前停止
                    min_delta=0.01,  # 损失改善的最小阈值为0.01
                    mode="min",  # 目标是最小化验证损失
                ),
                LearningRateMonitor(),
                MyModelCheckpoint(
                    save_top_k=2,
                    dirpath=config.model_save_dir,
                    monitor=BasicExpirement.VAL_RECONS_LOSS,
                    mode="min",
                    every_n_epochs=5,
                ),
            ],
        )
        logger.info("init trainer succeed")

        # start to train
        runner.fit(exp, datamodule=loader)
    except Exception as e:
        logger.info(f"{traceback.format_exc()}")
        exit(1)
