import config
import torch
import numpy as np


class BERT_KBQA_Dataloader_hard_neg:
    def __init__(self, text, relation_cat, mask_labels):
        self.text = text
        # self.text, self.relation, self.relation_cat = self.data_item.split('\t')
        self.relation1 = relation1
        self.relation2 = relation2
        self.relation3 = relation3
        self.relation4 = relation4
        self.relation5 = relation5
        self.mask_labels = mask_labels
        self.tokenizer = config.tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text= str(self.text[index])
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
            'targets1': torch.tensor(self.relation1[index], dtype=torch.long),
            'targets2': torch.tensor(self.relation2[index], dtype=torch.long),
            'targets3': torch.tensor(self.relation3[index], dtype=torch.long),
            'targets4': torch.tensor(self.relation4[index], dtype=torch.long),
            'targets5': torch.tensor(self.relation5[index], dtype=torch.long),
            'mask_labels': torch.from_numpy(np.array(self.mask_labels[index]))
        }
