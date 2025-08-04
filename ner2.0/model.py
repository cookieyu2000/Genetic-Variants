import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

class NERModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)