# dataprocess_text.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset   # HuggingFace datasets library
from transformers import AutoTokenizer
from text_configs import get_dataset_config, get_tokenizer_config, validate_dataset_config

class TextCLSDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased", dataset_name="ag_news", max_len=None):
        # Validate configurations
        validate_dataset_config(dataset_name, tokenizer_name)
        
        # Get dataset and tokenizer configs
        dataset_config = get_dataset_config(dataset_name)
        tokenizer_config = get_tokenizer_config(tokenizer_name)
        
        # Load dataset split
        self.dataset = load_dataset(dataset_config['name'], split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set max length (use config default if not specified)
        self.max_len = max_len if max_len is not None else tokenizer_config['max_len']
        
        # Use config field names
        self.text_field = dataset_config['text_field']
        self.label_field = dataset_config['label_field']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Use configurable field names
        text = self.dataset[idx][self.text_field]
        label = self.dataset[idx][self.label_field]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label)


def get_dataloaders(batch_size=32, dataset_name="ag_news", tokenizer_name="bert-base-uncased", 
                    max_len=None, num_workers=4):
    """Create dataloaders using configuration system
    
    Args:
        batch_size: Batch size for dataloaders
        dataset_name: Name of the dataset (from text_configs)
        tokenizer_name: Name of the tokenizer (from text_configs) 
        max_len: Maximum sequence length (uses config default if None)
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    train_ds = TextCLSDataset("train", tokenizer_name, dataset_name, max_len)
    test_ds  = TextCLSDataset("test",  tokenizer_name, dataset_name, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_dataset_info(dataset_name):
    """Get information about a dataset configuration
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict: Dataset configuration information
    """
    return get_dataset_config(dataset_name)


def get_available_datasets():
    """Get list of available dataset names"""
    from text_configs import get_available_datasets
    return get_available_datasets()


def get_available_tokenizers():
    """Get list of available tokenizer names"""
    from text_configs import get_available_tokenizers
    return get_available_tokenizers()
