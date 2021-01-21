from transformers import BertTokenizer

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
ACCUMULATION = 3
# BERT_PATH='../data/bert_base_uncased'
# MODEL_PATH='model.bin'
# TRAINING_FILE = '../data/train.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_name = "bert-base-uncased"
