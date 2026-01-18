import sys
import logging
import argparse

import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from tensorflow import keras
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.append(str(root_path))
from data.data_init import (
    get_tokenized_data,
    model_checkpoint,
    
    )
from set_logging import setup_logging
setup_logging()
logging = logging.getLogger(filename)



def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Enter arguments for learning'
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        default = 16
    )

    parser.add_argument(
        '--lr',
        type = float,
        default = 1e-02
    )

    parser.add_argument(
       '--epochs',
       type = int,
       default = 3 
    )

    return parser.parse_args()


def main():

    args = parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs

    model = TFBertForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    collator = DataCollatorWithPadding(tokenizer)


    train_data, val_data = get_tokenized_data()

    train_data = train_data.to_tf_dataset(
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collator,
        columns = ['input_ids', 'token_type_ids', 'attention_mask'],
        label_cols = ['label']
    )
    val_data = val_data.to_tf_dataset(
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collator,
        columns = ['input_ids', 'token_type_ids', 'attention_mask'],
        label_cols = ['label']
    )
    

if __name__ == '__main__':
    main()
    