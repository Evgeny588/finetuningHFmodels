import tensorflow as tf
import keras
import logging
import sys
from pathlib import Path
from tqdm.auto import tqdm

root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.path.append(str(root_path))

from set_logging import setup_logging

# Logging
setup_logging()
logger = logging.getLogger(filename)

def train_loop(model, optimizer, loss_fn, metric, train_data, verbose: bool = True):
    # Batch loop
    metric.reset_state()
    train_losses = 0.0
    train_batch_counter = 0

    for inputs, targets in tqdm(train_data, desc = 'Train loop', leave = False):
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        with tf.GradientTape() as tape:
            logits = model(
                input_ids = input_ids,
                attention_mask = attention_masks,
                token_type_ids = token_type_ids,
                training = True
            ).logits

            loss = loss_fn(
                targets,
                logits 
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric.update_state(targets, logits)
        
        train_losses += loss
        train_batch_counter += 1

    epoch_loss = train_losses / train_batch_counter
    epoch_accuracy = metric.result()
    
    if verbose:
        logger.info(f'Train_loss: {epoch_loss: .4f}')
        logger.info(f'Train_accuracy: {epoch_accuracy: .4f}')

    return epoch_loss, epoch_accuracy


def validation_loop(model, loss_fn, metric, validation_data, verbose: bool = True):
    # Batch loop
    metric.reset_state()
    eval_losses = 0.0
    eval_batch_counter = 0

    for inputs, targets in tqdm(validation_data, desc = 'Eval loop', leave = False):
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_masks = inputs['attention_mask']

        logits = model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_masks
        ).logits
        
        loss = loss_fn(
            targets,
            logits
        )

        metric.update_state(targets, logits)
        eval_losses += loss
        eval_batch_counter += 1 

    epoch_loss = eval_losses / eval_batch_counter
    epoch_accuracy = metric.result()

    if verbose:
        logger.info(f'Validation_loss: {epoch_loss: .4f}')
        logger.info(f'Validation_metric: {epoch_accuracy: .4f}')
        
    return epoch_loss, epoch_accuracy

