from transformers import BertTokenizer

MAX_LEN = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_name = "bert-base-uncased"
