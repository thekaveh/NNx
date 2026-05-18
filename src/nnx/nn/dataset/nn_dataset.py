from functools import reduce
from dataclasses import dataclass
from typing import Callable, Type, Tuple, Optional

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split

from .nn_dataset_base import NNDatasetBase

@dataclass(frozen=True, kw_only=True, slots=True)
class NNDataset(NNDatasetBase):
    ds_class        : Type[VisionDataset]
    root_dir        : str                   = "./data"
    download        : bool                  = True
    transform       : Optional[Callable]    = None
    batch_sizes     : Tuple[int, int, int]  = (None, None, None)
    val_proportion  : float                 = 0.1
    
    def __post_init__(self):        
        train_dataset, non_train_dataset = (
            self.ds_class(root=self.root_dir, train=True, download=self.download, transform=self.transform)
            , self.ds_class(root=self.root_dir, train=False, download=self.download, transform=self.transform)
        )
        
        val_dataset, test_dataset = random_split(
            non_train_dataset
            , [
                int(len(non_train_dataset) * self.val_proportion)
                , int(len(non_train_dataset) * (1 - self.val_proportion))
            ]
        )
        
        object.__setattr__(
            self
            , 'name'
            , self.ds_class.__name__
        )
        
        train_batch_size    = self.batch_sizes[0] or len(train_dataset)
        val_batch_size      = self.batch_sizes[1] or len(val_dataset)
        test_batch_size     = self.batch_sizes[2] or len(test_dataset)
        batch_sizes         = (train_batch_size, val_batch_size, test_batch_size)
        
        object.__setattr__(
            self
            , 'batch_sizes'
            , batch_sizes
        )

        object.__setattr__(
            self
            , 'train_loader'
            , DataLoader(
                shuffle=True
                , dataset=train_dataset
                , batch_size=self.batch_sizes[0]
            )
        )

        object.__setattr__(
            self
            , 'val_loader'
            , DataLoader(
                shuffle=False
                , dataset=val_dataset
                , batch_size=self.batch_sizes[1]
            )
        )

        object.__setattr__(
            self
            , 'test_loader'
            , DataLoader(
                shuffle=False
                , dataset=test_dataset
                , batch_size=self.batch_sizes[2]
            )
        )
        
        object.__setattr__(
            self
            , 'input_dim'
            , reduce(lambda x, y: x * y, self.train_loader.dataset[0][0].shape)
        )
        
        object.__setattr__(
            self
            , 'output_dim'
            , len(self.train_loader.dataset.classes)
        )
        
        state = dict(
            name                = self.name
            , input_dim         = self.input_dim
            , output_dim        = self.output_dim
            , train_batch_size  = f"{self.batch_sizes[0]:,}"
            , val_batch_size    = f"{self.batch_sizes[1]:,}"
            , test_batch_size   = f"{self.batch_sizes[2]:,}"
        )
        
        object.__setattr__(self, '_state', state)