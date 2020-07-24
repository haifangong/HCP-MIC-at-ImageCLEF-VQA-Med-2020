from bert.modeling_bert import *
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class Question_Classifier(nn.Module):
    def __init__(self, bert_mode, bert_pretrain, num_classes=3):
        super(Question_Classifier, self).__init__()
        self.q_emb = BertModel.from_pretrained(bert_pretrain)
        if bert_mode == 'base':
            q_dim = 768

        self.classifier = SimpleClassifier(q_dim, q_dim * 2, num_classes)

    def forward(self, x):
        q_emb = self.q_emb(x)
        out = self.classifier(q_emb.sum(1))
        return out
