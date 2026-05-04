import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output  # (batch, 768)

        logits = self.classifier(pooled_output)

        return logits