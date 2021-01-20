import config
import torch
import numpy as np


class BERT_KBQA_Dataloader:
    def __init__(self, text, relation_cat, mask_labels):
        self.text = text
        # self.text, self.relation, self.relation_cat = self.data_item.split('\t')
        self.relation_cat = relation_cat
        self.mask_labels = mask_labels
        self.tokenizer = config.tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text= str(self.text)
        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=config.MAX_LEN)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = config.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.relation_cat[index], dtype=torch.long),
            'mask_labels': torch.from_numpy(np.array(self.mask_labels[index]))
        }
