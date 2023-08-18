import json
from pathlib import Path

import numpy as np
from dgl import load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import SubsetRandomSampler
import torch

class GraphDataset(DGLDataset):
    def __init__(self,graphs,targets):
        super().__init__('GraphDataset')
        self.graphs = graphs
        self.targets = targets
    def __getitem__(self, idx): 
        return self.graphs[idx],self.targets[idx]
    def __len__(self):
        return len(self.graphs)
    

class PLGraphDataLoader(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.spath = cfg.train.spath
        self.target = cfg.target

        self.splitfilename = cfg.train.splitfilename
        self.load_and_split = cfg.train.load_and_split
        
        self.batch_size = cfg.train.batch_size 
        self.num_workers = cfg.num_workers
        self.sfilename = cfg.train.sfilename
        
        if self.load_and_split:
            self.train_split = cfg.train.train_split
            self.val_split = cfg.train.val_split
            self.test_split = cfg.train.test_split
        

    def setup(self, stage=None):

        if self.load_and_split: 
            assert self.train_split + self.val_split + self.test_split == 1
   
            graphs,target_dict =  load_graphs(str(Path(self.spath).joinpath(self.sfilename)))
            targets = target_dict[self.target]

            indices = list(range(len(graphs))) 
            np.random.shuffle(indices)
            train_size = int(self.train_split * len(graphs))
            val_size = train_size + int(self.val_split * len(graphs))

            tr_graphs = graphs[:train_size]
            tr_targets = targets[:train_size]

            val_graphs = graphs[train_size:val_size]
            val_targets = targets[train_size:val_size]

            tt_graphs = graphs[val_size:]
            tt_targets = targets[val_size:]
            tt_names = target_dict['graphs_ids'][val_size:] #list(range(len(tt_graphs))) #
            
            ids = {'train' : indices[:train_size],
                   'val' : indices[train_size:val_size],
                   'test' : indices[val_size:]}

        else:            
            ids = self._read_splits()     

            tr_graphs,target_dict = load_graphs(str(Path(self.spath).joinpath(self.sfilename)),idx_list=ids['train'])
            targets = target_dict[self.target]
            tr_targets = targets[ids['train']]
            
            val_graphs,_ = load_graphs(str(Path(self.spath).joinpath(self.sfilename)),idx_list=ids['val'])
            val_targets = targets[ids['val']]
            
            tt_graphs,_ = load_graphs(str(Path(self.spath).joinpath(self.sfilename)),idx_list=ids['test'])
            tt_targets = targets[ids['test']]
            tt_names = target_dict['graphs_ids'][ids['test']]

        self.train_data = GraphDataset(tr_graphs,tr_targets)
        self.val_data = GraphDataset(val_graphs,val_targets)
        self.test_data = GraphDataset(tt_graphs,list(zip(tt_targets,tt_names)))
        self._save_splits(ids)
                                                
    def train_dataloader(self):
        return GraphDataLoader(
            self.train_data, 
            sampler=SubsetRandomSampler(np.arange(len(self.train_data))), 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True, 
            drop_last=False
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.val_data, 
            sampler=SubsetRandomSampler(np.arange(len(self.val_data))), 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True, 
            drop_last=False
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_data, 
            sampler=SubsetRandomSampler(np.arange(len(self.test_data))), 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True, 
            drop_last=False
        )

    def _save_splits(self,ids):
        with open(Path(self.spath).joinpath(self.splitfilename),'w') as f:
            json.dump(ids,f)

    def _read_splits(self):
        with open(Path(self.spath).joinpath(self.splitfilename),'r') as f:
            return json.load(f)