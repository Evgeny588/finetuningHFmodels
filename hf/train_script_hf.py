import logging
import sys
import warnings
import os

from pathlib import Path
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.path.append(str(root_path))
warnings.filterwarnings('ignore')
os.environ['WANDB_DISABLED'] = 'true'
 
from data.data_init import (
    model_checkpoint,
    get_tokenized_data,
)
from set_logging import setup_logging

# Logging setups
setup_logging(console_level = logging.WARNING, file_mode = 'w')
logging = logging.getLogger(filename)


# Main code
def main():
    # Model initialize
    model = BertForSequenceClassification.from_pretrained(model_checkpoint)
    logging.debug(f'Model {model_checkpoint} was initialized.')

    # Other initializations
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    collator = DataCollatorWithPadding(tokenizer)

    # Data initialize
    train_data, val_data = get_tokenized_data()
    logging.debug('Train and validation data initialized.')

    # Arguments for train
    train_args = TrainingArguments(
        str(Path(root_path) / 'checkpoints'),
        eval_strategy = 'steps',
        eval_steps = 1000,
        logging_dir = str(Path(root_path) / 'logs'),
        report_to = []
    )

    # Trainer for model
    trainer = Trainer(
        model = model,
        args = train_args,
        data_collator = collator,
        train_dataset = train_data,
        eval_dataset = val_data,
        tokenizer = tokenizer
    )

    trainer.train()

if __name__ == '__main__':
    main()
