import torch_geometric as pyg

from dataclasses import dataclass
from torch_geometric.data import Dataset
from torch_geometric.loader import NeighborLoader
from typing import Callable, Type, Tuple, Optional, List

from .nn_dataset_base import NNDatasetBase

@dataclass(frozen=True, kw_only=True, slots=True)
class NNGraphDataset(NNDatasetBase):
    ds_class        : Type[Dataset]
    n_neighbors     : List[int]
    root_dir        : str                   = "./data"
    transform       : Optional[Callable]    = None
    n_workers       : int                   = 4
    batch_sizes     : Tuple[int, int, int]  = (None, None, None)
    
    def __post_init__(self):
        dataset = self.ds_class(root=self.root_dir, transform=self.transform)
        
        object.__setattr__(
            self
            , 'name'
            , self.ds_class.__name__
        )
        
        train_batch_size    = self.batch_sizes[0] or int(dataset._data.train_mask.sum())
        val_batch_size      = self.batch_sizes[1] or int(dataset._data.val_mask.sum())
        test_batch_size     = self.batch_sizes[2] or int(dataset._data.test_mask.sum())
        batch_sizes         = (train_batch_size, val_batch_size, test_batch_size)
        
        object.__setattr__(
            self
            , 'batch_sizes'
            , batch_sizes
        )
        
        object.__setattr__(
            self
            , 'train_loader'
            , NeighborLoader(
                shuffle=True
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , batch_size=self.batch_sizes[0]
                , input_nodes=dataset._data.train_mask
            )
        )
        
        object.__setattr__(
            self
            , 'val_loader'
            , NeighborLoader(
                shuffle=False
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , batch_size=self.batch_sizes[1]
                , input_nodes=dataset._data.val_mask
            )
        )
        
        object.__setattr__(
            self
            , 'test_loader'
            , NeighborLoader(
                shuffle=False
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , batch_size=self.batch_sizes[2]
                , input_nodes=dataset._data.test_mask
            )
        )
        
        object.__setattr__(
            self
            , 'input_dim'
            , dataset.num_features
        )
        
        object.__setattr__(
            self
            , 'output_dim'
            , dataset.num_classes
        )
        
        state = dict(
            name                = self.name
            , input_dim         = self.input_dim
            , output_dim        = self.output_dim
            , train_batch_size  = f"{self.batch_sizes[0]:,}"
            , val_batch_size    = f"{self.batch_sizes[1]:,}"
            , test_batch_size   = f"{self.batch_sizes[2]:,}"
            , n_workers         = self.n_workers
            , n_neighbors       = self.n_neighbors
        )
        
        object.__setattr__(self, '_state', state)