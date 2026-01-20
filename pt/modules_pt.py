import torch 
import logging
import sys

from torch import nn
from tqdm.auto import tqdm
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
filename = Path(__file__).stem

from set_logging import setup_logging

# Logging
setup_logging()
logger = logging.getLogger(filename)


def one_epoch(model, optimizer, loss_fn, train_loader, val_loader, device, epoch, validation_cycle = None, val_per_epoch = 1):     
    model.train() 
    # Cumulative variables
    current_loss = 0.0
    batch_counter = 0

    # Loop
    for inputs in tqdm(train_loader, desc = 'Train_loop', leave = False):
        inputs = {k: v.to(device) for k, v in inputs.items()}  
        
        # Forward pass
        loss = model(**inputs).loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Cumulative
        current_loss += loss.item()
        batch_counter += 1

        logger.debug(f'{batch_counter} train cycles is full') 

    train_epoch_loss = current_loss / batch_counter

    if (validation_cycle is not None) and (epoch % val_per_epoch == 0):
        val_epoch_loss = validation_cycle(
            model = model,
            loss_fn = loss_fn,
            val_loader = val_loader,
            device = device
        )
      
    return train_epoch_loss, val_epoch_loss


def validation_cycle(model, loss_fn, val_loader, device):
    model.eval()
    # Cummulative variables
    current_loss = 0.0
    batch_counter = 0

    # Loop
    for inputs in tqdm(val_loader, desc = 'Val_loop', leave = False):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Validation pass
        with torch.no_grad():
            loss = model(**inputs).loss

        # Cumulative
        current_loss += loss.item()
        batch_counter += 1

        logger.debug(f'{batch_counter} val cycles is full')
    return current_loss / batch_counter