import transformers
import torch.nn as nn
from config import model_name


class RDF_classifier_Model(nn.Module):
    def __init__(self):
        super(RDF_classifier_Model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.bert_drop = nn.Dropout(0.2)
        self.linear_layer = nn.Linear(768, 128)
        self.out = nn.Linear(128, 762)

    def forward(self, ids, mask, token_type_ids):
        _, out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bert_output = self.bert_drop(out)
        linear_out = self.linear_layer(bert_output)
        output_drop = self.bert_drop(linear_out)
        final_output = self.out(output_drop)
        return final_output
