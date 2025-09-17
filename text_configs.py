"""
Text dataset configurations for ANN-SNN conversion
"""

# Common text classification datasets configuration
TEXT_DATASETS = {
    'ag_news': {
        'name': 'ag_news',
        'text_field': 'text',
        'label_field': 'label',
        'num_classes': 4,
        'description': 'AG News topic classification dataset'
    },
    'imdb': {
        'name': 'imdb', 
        'text_field': 'text',
        'label_field': 'label',
        'num_classes': 2,
        'description': 'IMDB movie review sentiment analysis'
    },
    'sst2': {
        'name': 'sst2',
        'text_field': 'sentence',
        'label_field': 'label', 
        'num_classes': 2,
        'description': 'Stanford Sentiment Treebank 2 (sentence-level)'
    },
    
}

# Default tokenizer configurations
TOKENIZER_CONFIGS = {
    'bert-base-uncased': {
        'name': 'bert-base-uncased',
        'max_len': 128,
        'description': 'BERT base uncased model'
    },
    'bert-base-cased': {
        'name': 'bert-base-cased',
        'max_len': 128,
        'description': 'BERT base cased model'
    },
    'distilbert-base-uncased': {
        'name': 'distilbert-base-uncased',
        'max_len': 128,
        'description': 'DistilBERT base uncased model'
    },
    'roberta-base': {
        'name': 'roberta-base',
        'max_len': 128,
        'description': 'RoBERTa base model'
    }
}

def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset"""
    if dataset_name not in TEXT_DATASETS:
        available = list(TEXT_DATASETS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
    return TEXT_DATASETS[dataset_name]

def get_tokenizer_config(tokenizer_name):
    """Get configuration for a specific tokenizer"""
    if tokenizer_name not in TOKENIZER_CONFIGS:
        available = list(TOKENIZER_CONFIGS.keys())
        raise ValueError(f"Tokenizer '{tokenizer_name}' not found. Available tokenizers: {available}")
    return TOKENIZER_CONFIGS[tokenizer_name]

def get_available_datasets():
    """Get list of available dataset names"""
    return list(TEXT_DATASETS.keys())

def get_available_tokenizers():
    """Get list of available tokenizer names"""
    return list(TOKENIZER_CONFIGS.keys())

def validate_dataset_config(dataset_name, tokenizer_name):
    """Validate that dataset and tokenizer configurations exist"""
    if dataset_name not in TEXT_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    if tokenizer_name not in TOKENIZER_CONFIGS:
        raise ValueError(f"Invalid tokenizer: {tokenizer_name}")
    return True