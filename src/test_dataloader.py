import os
import hydra
from lib.lib_trainer import PL_EGAT
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
    model = PL_EGAT(cfg)
    for name,param in model.named_parameters():
        print(name,param.numel(),param.shape)
        tot_params+=param.numel()
    print("Tot params =",tot_params)
    dataloader = PLGraphDataLoader(cfg)
    dataloader.setup()
    print('Loaded data')
    g,_,y= next(iter(dataloader.train_dataloader()))
    print('Next OK')
    print(model(g)) 


    g,_,(y,name)= next(iter(dataloader.test_dataloader()))
    print(model(g))
    print(name)

def train(cfg):
    model = PL_EGAT(cfg)
    logger = CSVLogger(save_dir='testlog', name = "test_log",)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(
        deterministic=cfg.deterministic,
        accelerator="gpu",
        devices=1,
        max_epochs=100,
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
        g = dgl.rand_graph(10,30)
        g.ndata['node_type'] = torch.randint(0,8,(10,))
        g.edata['bond_dist_exp'] = torch.randn(30,100)
        self.g = g
    def __getitem__(self, index) -> Any:
        return self.g, torch.rand(1)
    def __len__(self):
        return 10

if __name__ == "__main__":
    main()
