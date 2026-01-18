import tensorflow as tf
import keras
import logging
import sys
from pathlib import Path

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

    for inputs, targets in train_data:
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
        train_batch_counter += input_ids.shape[0]

    epoch_loss = train_losses / train_batch_counter
    epoch_accuracy = metric.result()
    
    if verbose:
        logger.info(f'Loss: {epoch_loss: .4f}')
        logger.info(f'Accuracy: {epoch_accuracy: .4f}')

    return epoch_loss, epoch_accuracy