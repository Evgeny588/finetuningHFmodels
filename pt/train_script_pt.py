import torch
import os
import sys
import logging
import argparse
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
from pt.modules_pt import one_epoch, validation_cycle


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
        '--epochs',
        type = int,
        default = 3
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



# Setup logging
setup_logging()
logger = logging.getLogger(str(filename))

def main():
    # Parse and set some arguments
    args = parse_args()
    epochs = args.epochs
    device = args.device
    num_workers = 0 if device != 'cuda' else os.cpu_count()
    pin_memory = num_workers != 0

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

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses, val_losses = [], []
    losses = dict()
    for epoch in tqdm(range(1, epochs + 1), desc = 'Epoch'):
        train_loss, val_loss = one_epoch(
            model = model,
            optimizer = optimizer,
            loss_fn = criterion,
            train_loader = train_loader,
            val_loader = val_loader,
            device = device,
            epoch = epoch,
            validation_cycle = validation_cycle
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
         
    losses['Train'] = train_losses
    losses['Val'] = val_losses
    torch.save(losses, Path(root_path) / 'checkpoints/losses.pt')

