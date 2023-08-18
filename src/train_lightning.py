import os
import time
from pathlib import Path
from typing import *
import hydra
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,LearningRateMonitor)
from pytorch_lightning.loggers import WandbLogger
from  datetime import datetime
from lib.lib_trainer import * 
from lib.lib_data import * 

def write_results_yaml(cfg: dict, data: dict = None):
    Path(cfg.train.dpath).mkdir(exist_ok=True, parents=True)
    if data is None:
        train_data = {
            "target": cfg.target,
            "num_epochs": cfg.train.num_epochs,
            "learning_rate": cfg.train.base_lr,
            "batch_size": cfg.train.batch_size,
            "dataset": cfg.train.spath,
        }
        with open(
            str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_train_results.yaml")), "w"
        ) as outfile:
            yaml.dump(train_data, outfile)
    else:
        with open(
            str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_train_results.yaml")), "a"
        ) as outfile:
            yaml.dump(data, outfile)


def get_model_name(model: GrapheNet2):
    raw_name = str(type(model.net))
    chars_to_remove = "<>'"
    translate_table = str.maketrans("", "", chars_to_remove)
    name = raw_name.translate(translate_table)

    return name.split(".")[-1]

def set_logger(cfg):
    now = datetime.now()
    now_string = now.strftime("%b-%d-%Y_%H:%M:%S")

    logger = WandbLogger(
        project=cfg.logger.project_name,
        save_dir = cfg.logger.logdir,
        log_model=True,
        name=f"{cfg.target}_{now_string}",
    )

    return logger#, f"{cfg.target}_{now_string}"

@hydra.main(version_base="1.2",config_path="config", config_name="train")
def main(cfg):

    os.environ.setdefault('DGLBACKEND','pytorch')
    seed_everything(42, workers=True)

    model = GrapheNet2(cfg)
    if cfg.train.compile:
        pass 
        # model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.dpath,
        save_top_k=1,
        monitor="val_loss",
        filename="best_loss_{val_loss:.5f}_{epoch}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=35,min_delta=1e-5, verbose=True, check_on_train_epoch_end=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    dataloaders = PLGraphDataLoader(cfg)

    wandb_logger = set_logger(cfg)

    trainer = Trainer(
        deterministic=cfg.deterministic,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.num_epochs,
        callbacks=[
            checkpoint_callback,
            model.get_progressbar(),
            early_stopping,
            lr_monitor,
        ],
        logger=wandb_logger
    )

    write_results_yaml(cfg)
    write_results_yaml(cfg, data={"model_name": get_model_name(model)})

    start = time.time()
    trainer.fit(model, dataloaders)
    end = time.time()

    print(
        f"Completed training:\n TARGET = {cfg.target}\n DATASET = {cfg.train.spath}\n NUM EPOCHS = {cfg.train.num_epochs}\n TRAINING TIME = {(end-start)/60:.3f} minutes"
    )

    write_results_yaml(cfg, data={"training_time": float((end - start) / 60)})

if __name__ == "__main__":
    main()