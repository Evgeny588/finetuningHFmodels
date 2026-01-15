from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

data_checkpoint = 'glue', 'sst2'
model_checkpoint = 'bert-base-uncased'

dataset = load_dataset(*data_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_func(batch):
    global library
    return tokenizer(
        batch['sentence'],
        truncation = True,
        padding = False,
    )

def get_tokenized_data():
    train = dataset['train']
    val = dataset['validation']

    tokenized_train = train.map(tokenize_func, remove_columns = ['sentence'], load_from_cache_file = False) 
    tokenized_val = val.map(tokenize_func, remove_columns = ['sentence'], load_from_cache_file = False)
    return tokenized_train, tokenized_val

def main():
    train_data, val_data = get_tokenized_data()
    print("Sample from train_data:", train_data[0])

if __name__ == '__main__':
    main()