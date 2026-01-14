from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

data_checkpoint = 'mteb/RuToxicOKMLCUPClassification'
model_checkpoint = 'DeepPavlov/rubert-base-cased'

dataset = load_dataset(data_checkpoint)

train_data = dataset['train']
test_data = dataset['test']

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BertForSequenceClassification.from_pretrained(model_checkpoint)
print(model)