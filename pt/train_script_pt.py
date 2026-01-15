import torch
import os
import sys
import logging
import argparse
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.path.append(str(root_path))

from set_logging import setup_logging
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from data.data_init import (
    model_checkpoint,
    get_tokenized_data
)


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Use for changes hyperparameters'
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        default = 16
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 1e-5
    )
    parser.add_argument(
        '--device',
        type = str,
        default = 'cuda'
    )
    return parser.parse_args()

args = parse_args()
device = args.device
num_workers = 0 if device != 'cuda' else os.cpu_count()
pin_memory = num_workers != 0


# Setup logging
setup_logging()
logging = logging.getLogger(str(filename))

def main():
    # Init model
    model = BertForSequenceClassification.from_pretrained(model_checkpoint)

    # Init common classes
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    collator = DataCollatorWithPadding(tokenizer)

    # Get data
    train_data, val_data = get_tokenized_data() 

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size = args.bs,
        shuffle = False,
        num_workers = num_workers,
        collate_fn = collator,
        pin_memory = pin_memory
        )
    val_loader = DataLoader(
        dataset = val_data,
        batch_size = args.bs,
        shuffle = False,
        num_workers = num_workers,
        collate_fn = collator,
        pin_memory = pin_memory
    )