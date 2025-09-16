# dataprocess_text.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset   # HuggingFace datasets
from transformers import AutoTokenizer

class TextCLSDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased",
                 max_len=128, dataset_name="ag_news"):
        # Load dataset split
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      item = self.dataset[idx]  # This is a dict from HuggingFace dataset
      text = item['text']
      label = item['label']

      # Tokenize text
      encoding = self.tokenizer(
          text,
          truncation=True,
          max_length=self.max_len,
          return_tensors='pt'
      )

      return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label)


def get_dataloaders(batch_size=32, max_len=128, dataset_name="ag_news",
                    tokenizer_name="bert-base-uncased", num_workers=4):
    train_ds = TextCLSDataset("train", tokenizer_name, max_len, dataset_name)
    test_ds  = TextCLSDataset("test",  tokenizer_name, max_len, dataset_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
