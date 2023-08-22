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
from matgl.layers import BondExpansion
from .scheduler import CyclicCosineDecayLR
from .models import *

class PLEGATBase(pl.LightningModule): 

    def __init__(self,cfg,**kwargs):

        super(PLEGATBase,self).__init__()
        
        # Training Hyperparameters
        self.target = cfg.target
        self.num_epochs = cfg.train.num_epochs
        self.batch_size = cfg.train.batch_size
        self.normalize = cfg.train.normalize
        self.do_bond_expansion = cfg.train.do_bond_expansion

        # Optim
        self.scheduler = cfg.optim.scheduler.name
        self.scheduler_args = cfg.optim.scheduler.args
        self.optimizer = cfg.optim.optimizer.name
        self.optimizer_args = cfg.optim.optimizer.args

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
        classifier_attn_heads = cfg.model.classifier_attn_heads
        is_graph_classification = cfg.model.is_graph_classification

        self.net = AllInOneEGAT(num_node_types,
                          node_embed_dim,
                          edge_embed_dim,
                          edge_dims,attn_dims,
                          attn_heads,
                          classifier_attn_heads,
                          niters_set2set,
                          nlayers_set2set,
                          hidden_layer_sizes_output,
                          is_graph_classification)
          
        self.save_hyperparameters(cfg)

    def forward(self, graph, node_feat,edge_feat):
        return self.net(graph,node_feat,edge_feat)
    
    def configure_optimizers(self):
        
        if self.optimizer == 'adam':
            opt = torch.optim.Adam(
                self.parameters(),
                **self.optimizer_args
                )        
        else: raise NotImplementedError('Ops!')
        

        if self.scheduler == 'cosine-annealing-with-restarts':
            scheduler = CyclicCosineDecayLR(
                opt, 
                **self.scheduler_args)


        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

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
    
class PLEGATRegressor(PLEGATBase): 

    def __init__(self,cfg,**kwargs):

        super(PLEGATRegressor,self).__init__(cfg,**kwargs)

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

    def forward(self,g,n,e):
        return super(PLEGATRegressor,self).forward(g,n,e)['graph']

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
        node_feat = g.ndata['node_type']
        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        y_hat = self(g,node_feat,edge_feat)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        mae = self.mae(y_hat, y)

        self.train_loss_step_holder.append(loss)
        self.train_acc_step_holder.append(acc)
        self.train_mae_step_holder.append(mae)

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        g, y  = val_batch
        node_feat = g.ndata['node_type']
        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']
        y_hat = self(g,node_feat,edge_feat)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        mae = self.mae(y_hat, y)

        self.val_loss_step_holder.append(loss)
        self.val_acc_step_holder.append(acc)
        self.val_mae_step_holder.append(mae)

        return loss

    def test_step(self, test_batch, batch_idx=None):
        g,(y,ids) = test_batch
        node_feat = g.ndata['node_type']
        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']
        y_hat = self(g,node_feat,edge_feat)
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

class PLEGATNodePredictor(PLEGATBase): 

    def __init__(self,cfg,**kwargs):

        super(PLEGATNodePredictor,self).__init__(cfg,**kwargs)

        self.count = 0
        self.accs = []
        self.sample_ids = []

        self.train_loss_step_holder = []
        self.train_acc_step_holder = []

        self.val_loss_step_holder = []
        self.val_acc_step_holder = []

        self.min_val_loss = float("inf")

        self.sample_frac = 0.2

    def forward(self,g,n,e):
        return super(PLEGATNodePredictor,self).forward(g,n,e)['node']

    def criterion(self, logits, labels):
        labels = labels.long()
        ce = nn.CrossEntropyLoss()
        return ce(logits,labels)
    
    def accuracy(self, logits, labels, test_step=False):
        labels = labels.long()
        pred = logits.argmax(1)
        acc = (pred == labels).float().mean()

        if test_step:
            return acc, pred
        else:
            return acc
        
    def training_step(self, train_batch, batch_idx=None, optimizer_idx=None):
        g, _ = train_batch
        node_types = g.ndata['node_type']
        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        logits = self(g,node_feat,edge_feat)

        loss = self.criterion(logits[mask], node_types[mask])
        acc = self.accuracy(logits[mask], node_types[mask])

        self.train_loss_step_holder.append(loss)
        self.train_acc_step_holder.append(acc)

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        g, _ = val_batch

        node_types = g.ndata['node_type']

        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        logits = self(g,node_feat,edge_feat)

        loss = self.criterion(logits[mask], node_types[mask])
        acc = self.accuracy(logits[mask], node_types[mask])

        self.val_loss_step_holder.append(loss)
        self.val_acc_step_holder.append(acc)

        return loss

    def test_step(self, test_batch, batch_idx=None):
        g,(_,ids) = test_batch
        node_types = g.ndata['node_type']
        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        logits = self(g,node_feat,edge_feat)
        acc,predictions = self.accuracy(logits[mask], node_types[mask],test_step=True)

        self.accs = [*self.accs, *acc.tolist()]
        self.plot_y = [*self.plot_y, *node_types.tolist()]
        self.plot_y_hat = [*self.plot_y_hat, *predictions.tolist()]

        self.sample_ids = [*self.sample_ids, *ids.cpu().numpy()]

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.train_acc_step_holder).mean(dim=0)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.train_loss_step_holder.clear()
        self.train_acc_step_holder.clear()

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.val_acc_step_holder).mean(dim=0)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.count += 1
        if self.min_val_loss > loss:
            print(
                f"In epoch {self.current_epoch} reached a new minimum for validation loss: {loss}, patience: {self.count} epochs"
            )
            self.min_val_loss = loss
            self.count = 0
            
        self.val_loss_step_holder.clear()
        self.val_acc_step_holder.clear()
            
    def on_train_start(self):
        self.log_dict(
            {
                "hp/num_epochs": float(self.num_epochs),
                "hp/batch_size": float(self.batch_size),
            }
        )

    def on_test_start(self):
        self.accs.clear()
        self.plot_y.clear()
        self.plot_y_hat.clear()
        self.sample_ids.clear()

    def mask_labels(self,labels):
        mask = torch.rand(len(labels),device=labels.device) < self.sample_frac
        return labels * ~mask, mask 