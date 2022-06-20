import torch
from torch.utils.data import Dataset, DataLoader


class RxnDatasetMLM(Dataset):
    def __init__(self,
                 file_path,
                 tokenizer,
                 ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        self.rxn_smiles = [self.preprocess(line) for line in lines]
    
    def preprocess(self, line):
        """Proprocess and tokenize the reaction smiles"""
        return self.tokenizer(line.strip(), truncation=True, padding='max_length')
    
    def __len__(self):
        return len(self.rxn_smiles)

    def __getitem__(self, index):
        return self.rxn_smiles[index]


def construct_mlm_loader(file_path, tokenizer, args, modes=('train', 'val')):
    """Create PyTorch DataLoader"""

    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        dataset = RxnDatasetMLM(file_path, tokenizer)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            )
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
