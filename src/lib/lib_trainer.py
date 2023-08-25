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

    def criterion(self, output, target):
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
        mask = torch.zeros(1)
        while mask.sum() == 0:
            mask = torch.rand(len(labels),device=labels.device) < self.sample_frac
        return labels * ~mask, mask 
    
class PLEGATNodePredictorAndRegressor(PLEGATBase): 

    def __init__(self,cfg,**kwargs):

        super(PLEGATNodePredictorAndRegressor,self).__init__(cfg,**kwargs)

        self.count = 0
        self.node_accs = []
        self.sample_ids = []


        self.min_val_loss = float("inf")

        self.sample_frac = 0.2

        self.reg_errors = []
        self.reg_maes = []
        self.reg_plot_y = []
        self.reg_plot_y_hat = []

        self.train_loss_step_holder = []
        self.train_node_acc_step_holder = []
        self.train_property_acc_step_holder = []
        self.train_property_mae_step_holder = []

        self.val_loss_step_holder = []
        self.val_node_acc_step_holder = []
        self.val_property_acc_step_holder = []
        self.val_property_mae_step_holder = []

        self.test_node_acc = []
        self.test_error_abs = []
        self.test_error_rel = []
        self.test_plot_y = []
        self.test_plot_y_hat = []
        self.test_sample_ids = []

        self.min_val_loss = float("inf")

    def criterion(self,output,node_labels,mask,target):
        ce = nn.CrossEntropyLoss()
        l2 = nn.MSELoss()

        logits = output['node'][mask]
        node_labels = node_labels.long()[mask]

        y_hat = torch.squeeze(output['graph'])
        if self.normalize == "z_score":
            target = (target - self.mean) / self.std
        elif self.normalize == "normalization":
            target = (target - self.min) / (self.max - self.min)
        elif self.normalize == "log":
            target = torch.log(target + (self.min + 1))
        
        return (ce(logits,node_labels) + (torch.sqrt(l2(y_hat, target))
                                      if self.target == "total_energy"
                                      else l2(y_hat, target))
                                      )
    
    def metrics(self, output,node_labels,mask,target, test_step=False):
        
        logits = output['node'][mask]
        node_labels = node_labels.long()[mask]
        pred = logits.argmax(1)

        y_hat = torch.squeeze(output['graph'])
        if self.normalize == "z_score":
            y_hat = (y_hat * self.std) + self.mean
        elif self.normalize == "normalization":
            y_hat = y_hat * (self.max - self.min) + self.min
        elif self.normalize == "log":
            y_hat = torch.exp(y_hat) - (self.min + 1)


        acc = (pred == node_labels).float().mean()
        error = torch.abs(y_hat - target) 

        error = torch.abs(y_hat - target)
       
        if test_step:
            return {'node_acc' : acc,
                    'property_error_rel' : error / torch.abs(target) * 100.0,
                    'property_error_abs' : error
                    }, output
        else:
            return {'node_acc' : acc,
                    'property_error_rel' : torch.mean(100.0 - error / torch.abs(target) * 100.0),
                    'property_error_abs' : torch.mean(error)
                    }
        
    def training_step(self, train_batch, batch_idx=None, optimizer_idx=None):
        g, y = train_batch
        node_types = g.ndata['node_type']
        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        pred = self(g,node_feat,edge_feat)

        loss = self.criterion(pred,node_types,mask,y)
        metrics = self.metrics(pred,node_types,mask,y)

        self.train_loss_step_holder.append(loss)
        self.train_node_acc_step_holder.append(metrics['node_acc'])
        self.train_property_acc_step_holder.append(metrics['property_error_rel'])
        self.train_property_mae_step_holder.append(metrics['property_error_abs'])
        return loss

    def validation_step(self, val_batch, batch_idx=None):
        g, y = val_batch
        node_types = g.ndata['node_type']
        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        pred = self(g,node_feat,edge_feat)
        
        loss = self.criterion(pred,node_types,mask,y)
        metrics = self.metrics(pred,node_types,mask,y)

        self.val_loss_step_holder.append(loss)
        self.val_node_acc_step_holder.append(metrics['node_acc'])
        self.val_property_acc_step_holder.append(metrics['property_error_rel'])
        self.val_property_mae_step_holder.append(metrics['property_error_abs'])
        return loss

    def test_step(self, test_batch, batch_idx=None):
        g,(y,ids) = test_batch
        node_types = g.ndata['node_type']
        node_feat,mask = self.mask_labels(node_types)

        if self.do_bond_expansion: 
            edge_feat = self.net.bond_expansion(g.edata['bond_dist'])
        else: 
            edge_feat = g.edata['bond_dist_exp']

        pred = self(g,node_feat,edge_feat)
        metrics,predictions = self.metrics(pred,node_types,mask,y,test_step=True)

        self.test_node_acc.clear()
        self.test_error_abs.clear()
        self.test_error_rel.clear()
        self.test_plot_y.clear()
        self.test_plot_y_hat.clear()
        self.test_sample_ids.clear()

        self.test_node_acc = [*self.test_node_acc, *metrics['node_acc'].tolist()]
        self.test_error_abs = [*self.test_error_abs, *metrics['property_error_rel'].tolist()]
        self.test_error_rel = [*self.test_error_rel, *metrics['property_error_abs'].tolist()]
        self.test_plot_y = [*self.test_plot_y, *predictions['graph'].squeeze().tolist()]
        self.test_sample_ids = [*self.test_sample_ids, *ids.cpu().numpy()]

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_loss_step_holder).mean(dim=0)
        node_acc = torch.stack(self.train_node_acc_step_holder).mean(dim=0)
        property_acc = torch.stack(self.train_property_acc_step_holder).mean(dim=0)
        property_mae = torch.stack(self.train_property_mae_step_holder).mean(dim=0)
        
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_node_acc", node_acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_property_acc", property_acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "train_property_mae", property_mae, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.train_loss_step_holder.clear()
        self.train_node_acc_step_holder.clear()
        self.train_property_acc_step_holder.clear()
        self.train_property_mae_step_holder.clear()

    def on_validation_epoch_end(self):

        loss = torch.stack(self.val_loss_step_holder).mean(dim=0)
        node_acc = torch.stack(self.val_node_acc_step_holder).mean(dim=0)
        property_acc = torch.stack(self.val_property_acc_step_holder).mean(dim=0)
        property_mae = torch.stack(self.val_property_mae_step_holder).mean(dim=0)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_node_acc", node_acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_property_acc", property_acc, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )
        self.log(
            "val_property_mae", property_mae, on_epoch=True, prog_bar=True, logger=True, on_step=False,sync_dist=True
        )

        self.count += 1
        if self.min_val_loss > loss:
            print(
                f"In epoch {self.current_epoch} reached a new minimum for validation loss: {loss}, patience: {self.count} epochs"
            )
            self.min_val_loss = loss
            self.count = 0
            
        self.val_loss_step_holder.clear()
        self.val_node_acc_step_holder.clear()
        self.val_property_acc_step_holder.clear()
        self.val_property_mae_step_holder.clear()
            
    def on_train_start(self):
        self.log_dict(
            {
                "hp/num_epochs": float(self.num_epochs),
                "hp/batch_size": float(self.batch_size),
            }
        )

    def on_test_start(self):
        self.test_node_acc.clear()
        self.test_error_abs.clear()
        self.test_error_rel.clear()
        self.test_plot_y.clear()
        self.test_plot_y_hat.clear()
        self.test_sample_ids.clear()


    def mask_labels(self,labels):
        mask = torch.zeros(1)
        while mask.sum() == 0:
            mask = torch.rand(len(labels),device=labels.device) < self.sample_frac
        return labels * ~mask, mask 