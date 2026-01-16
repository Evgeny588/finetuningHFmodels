import sys
import logging


import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification
)
from tensorflow import keras
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.append(str(root_path))
from data.data_init import (
    get_tokenized_data,
    model_checkpoint
    )


def main():
    train_data, val_data = get_tokenized_data()

    train_data = train_data.to_tf_dataset()
    val_data = val_data.to_tf_dataset()

    model = TFBertForSequenceClassification.from_pretrained(model_checkpoint)