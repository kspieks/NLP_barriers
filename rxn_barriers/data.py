import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TorchStandardScaler(torch.nn.Module):
    """
    Standard Scaler Class to z-score data
    
    Args:
        eps: tolerance to avoid dividing by 0.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def fit(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def transform(self, x):
        x = x - self.mean
        x = x / (self.std + self.eps)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.std + self.eps)
        x = x + self.mean
        return x


class RxnDatasetMLM(Dataset):
    def __init__(self,
                 file_path,
                 tokenizer,
                 ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        self.encodings = [self.preprocess(line) for line in lines]
    
    def preprocess(self, smi):
        """
        Proprocess and tokenize the reaction smiles

        Args:
            smi: string representing the reaction SMILES

        Returns:
            tokenized_smi: dictionary with keys `input_ids`, `token_type_ids`, `attention_mask`.
        """
        return self.tokenizer(smi.strip(), truncation=True, padding='max_length')
    
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def construct_mlm_loader(tokenizer, args, modes=('train', 'val')):
    """Create PyTorch DataLoader"""

    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        dataset = RxnDatasetMLM(args.mlm_train_path if mode == 'train' else args.mlm_eval_path,
                                tokenizer,
                                )
        loader = DataLoader(dataset=dataset,
                            batch_size=args.per_device_train_batch_size if mode == 'train' else args.per_device_eval_batch_size,
                            shuffle=True if mode == 'train' else False,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=args.dataloader_pin_memory,
                            )
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders

class RxnDatasetRegression(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 targets,
                 ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.targets = targets
        
        self.df = pd.read_csv(self.data_path)
        self.encodings = [self.preprocess(smi) for smi in self.df.rxn_smiles]

        self.labels = self.get_targets()
        
    @property
    def mean(self):
        return self.labels.mean(dim=0, keepdim=True)

    @property
    def std(self):
        return self.labels.std(dim=0, unbiased=False, keepdim=True)

    def get_targets(self):
        """Create list of targets for regression"""
        return torch.tensor(self.df[self.targets].values)

    def preprocess(self, smi):
        """
        Proprocess and tokenize the reaction smiles

        Args:
            smi: string representing the reaction SMILES

        Returns:
            tokenized_smi: dictionary with keys `input_ids`, `token_type_ids`, `attention_mask`.
        """
        return self.tokenizer(smi.strip(), truncation=True, padding='max_length')
    
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # convert values from list to tensor
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = self.labels[idx]
        return item
