from pathlib import Path
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
import json

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from torch import nn
from matgl.models import MEGNet
from matgl.layers import BondExpansion
from .scheduler import CyclicCosineDecayLR
from .models import *

class PL_EGAT(pl.LightningModule): 

    def __init__(self,cfg,**kwargs):

        super(PL_EGAT,self).__init__()
        
        # Training Hyperparameters
        self.learning_rate = cfg.train.base_lr 
        self.target = cfg.target
        self.num_epochs = cfg.train.num_epochs
        self.batch_size = cfg.train.batch_size
        self.normalize = cfg.train.normalize
        self.do_bond_expansion = cfg.train.do_bond_expansion

        self.count = 0
        self.errors = []
        self.maes = []
        self.plot_y = []
        self.plot_y_hat = []
        self.sample_ids = []

        self.train_loss_step_holder = []
        self.train_acc_step_holder = []
        self.train_mae_step_holder = []

        self.val_loss_step_holder = []
        self.val_acc_step_holder = []
        self.val_mae_step_holder = []

        self.min_val_loss = float("inf")

        # Loss adjustment
        infopath = Path(cfg.train.spath).joinpath(cfg.train.infofile)
        if self.normalize  == "z_score":
            with open(infopath,'r') as f: 
                data = json.load(f)
            self.mean = data[cfg.target]['mean']
            self.std = data[cfg.target]['std']

        elif self.normalize == "normalization": 
            with open(infopath,'r') as f: 
                data = json.load(f)
            self.min = data[cfg.target]['min']
            self.max = data[cfg.target]['max']

        elif self.normalize == "log":
            with open(infopath,'r') as f: 
                data = json.load(f)
            self.minimun = np.abs(data[cfg.target]['min']) #FIXME don't know the difference between min and minimum, if any

        # Network parameters
        num_node_types = cfg.model.num_node_types
        node_embed_dim = cfg.model.node_embed_dim
        edge_embed_dim = cfg.model.edge_embed_dim
        edge_dims = cfg.model.edge_dims
        attn_dims = cfg.model.attn_dims
        attn_heads = cfg.model.attn_heads
        niters_set2set = cfg.model.niters_set2set
        nlayers_set2set = cfg.model.nlayers_set2set
        hidden_layer_sizes_output = cfg.model.hidden_layer_sizes_output
        self.net = MyEGATRegressor(num_node_types,
                          node_embed_dim,
                          edge_embed_dim,
                          edge_dims,attn_dims,
                          attn_heads,
                          niters_set2set,
                          nlayers_set2set,
                          hidden_layer_sizes_output)
          
        self.save_hyperparameters(cfg)

    def forward(self, graph):
        node_feat = graph.ndata['node_type']
        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(graph.edata['bond_dist'])
        else: 
            edge_feat = graph.edata['bond_dist_exp']

        out = self.net(graph,node_feat,edge_feat)

        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            )        
    
        scheduler = CyclicCosineDecayLR(
            opt, 
            init_decay_epochs=100,
            min_decay_lr=self.learning_rate * 0.1,
            restart_interval = 30,
            restart_interval_multiplier = None,
            restart_lr=self.learning_rate * 0.7,
            warmup_epochs = 30,
            warmup_start_lr = self.learning_rate * 0.3,
            last_epoch = -1,
            verbose = False)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def criterion(self, output, target, data=None):
        l2 = nn.MSELoss()

        output = torch.squeeze(output)

        if self.normalize == "z_score":
            target = (target - self.mean) / self.std
        elif self.normalize == "normalization":
            target = (target - self.min) / (self.max - self.min)
        elif self.normalize == "log":
            target = torch.log(target + (self.min + 1))
        
        return (
            torch.sqrt(l2(output, target))
            if self.target == "total_energy"
            else l2(output, target)
        )
    
    def mae(self, output, target, test_step=False):
        
        output = torch.squeeze(output)
        
        if self.normalize == "z_score":
            output = (output * self.std) + self.mean
        elif self.normalize == "normalization":
            output = output * (self.max - self.min) + self.min
        elif self.normalize == "log":
            output = torch.exp(output) - (self.min + 1)

        error = torch.abs(output - target)

        if test_step:
            return error
        else:
            return torch.mean(error)
        
    def accuracy(self, output, target, test_step=False):
        
        output = torch.squeeze(output)
        
        if self.normalize == "z_score":
            output = (output * self.std) + self.mean
        elif self.normalize == "normalization":
            output = output * (self.max - self.min) + self.min
        elif self.normalize == "log":
            output = torch.exp(output) - (self.min + 1)

        error = torch.abs(output - target) / torch.abs(target) * 100.0

        if test_step:
            return error, output
        else:
            return torch.mean(100.0 - error)
        
    def training_step(self, train_batch, batch_idx=None, optimizer_idx=None):
        g, y = train_batch
        y_hat = self(g)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        mae = self.mae(y_hat, y)

        self.train_loss_step_holder.append(loss)
        self.train_acc_step_holder.append(acc)
        self.train_mae_step_holder.append(mae)

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        g, y  = val_batch
        y_hat = self(g)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        mae = self.mae(y_hat, y)

        self.val_loss_step_holder.append(loss)
        self.val_acc_step_holder.append(acc)
        self.val_mae_step_holder.append(mae)

        return loss

    def test_step(self, test_batch, batch_idx=None):
        g,(y,ids) = test_batch
        y_hat = self(g)
        error, predictions = self.accuracy(y_hat, y, test_step=True)
        mae = self.mae(y_hat, y, test_step=True)
        self.errors = [*self.errors, *error.tolist()]
        self.maes = [*self.maes, *mae.tolist()]
        self.plot_y = [*self.plot_y, *y.tolist()]
        self.plot_y_hat = [*self.plot_y_hat, *predictions.tolist()]

        self.sample_ids = [*self.sample_ids, *ids.cpu().numpy()]

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.train_acc_step_holder).mean(dim=0)
        mae = torch.stack(self.train_mae_step_holder).mean(dim=0)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_mae", mae, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.train_loss_step_holder.clear()
        self.train_mae_step_holder.clear()
        self.train_acc_step_holder.clear()

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.val_acc_step_holder).mean(dim=0)
        mae = torch.stack(self.val_mae_step_holder).mean(dim=0)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_mae", mae, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.count += 1
        if self.min_val_loss > loss:
            print(
                f"In epoch {self.current_epoch} reached a new minimum for validation loss: {loss}, patience: {self.count} epochs"
            )
            self.min_val_loss = loss
            self.count = 0
            
        self.val_loss_step_holder.clear()
        self.val_mae_step_holder.clear()
        self.val_acc_step_holder.clear()
            
            

    def on_train_start(self):
        self.log_dict(
            {
                "hp/num_epochs": float(self.num_epochs),
                "hp/batch_size": float(self.batch_size),
            }
        )

    def on_test_start(self):
        self.errors.clear()
        self.maes.clear()
        self.plot_y.clear()
        self.plot_y_hat.clear()
        self.sample_ids.clear()

    @staticmethod
    def get_progressbar():
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="#e809a1",
                progress_bar="#6206E0",
                progress_bar_finished="#00c900",
                progress_bar_pulse="#6206E0",
                batch_progress="#e809a1",
                time="#e8c309",
                processing_speed="#e8c309",
                metrics="#dbd7d7",
            )
        )

        return progress_bar
