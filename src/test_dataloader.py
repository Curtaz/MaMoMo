import os
import hydra
from lib.lib_trainer import *
from lib.lib_data import PLGraphDataLoader
import dgl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, SequentialSampler
from dgl.dataloading import GraphDataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger,WandbLogger
import torch
from typing import Any


@hydra.main(version_base="1.2",config_path="config", config_name="train")
def main(cfg):

    os.environ.setdefault('DGLBACKEND','pytorch')
    tot_params = 0
    model = PLEGATNodePredictor(cfg)
    print(model)
    for name,param in model.named_parameters():
        print(name,param.numel(),param.shape)
        tot_params+=param.numel()
    print("Tot params =",tot_params)
    dataset = BruttoDataset()
    g,y = dataset[0]
    print(model(g,g.ndata['node_type'],g.edata['bond_dist_exp'])) 

    train(cfg)

def train(cfg):
    model = PLEGATNodePredictor(cfg)
    logger = CSVLogger(save_dir='testlog', name = "test_log",)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(
        deterministic=cfg.deterministic,
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        callbacks=[
            model.get_progressbar(),
            lr_monitor,
        ],
        logger=logger
    )
    dataset = BruttoDataset()
    trainer.fit(model,train_dataloaders=GraphDataLoader(dataset,sampler = SequentialSampler(dataset)))


class BruttoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        g = dgl.rand_graph(10,80)
        g.ndata['node_type'] = torch.randint(0,8,(10,))
        g.edata['bond_dist_exp'] = torch.randn(80,100)
        self.g = g
    def __getitem__(self, index) -> Any:
        return self.g, torch.rand(1)
    def __len__(self):
        return 10

if __name__ == "__main__":
    main()
