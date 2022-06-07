import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig


class BERT(nn.Module):

    def __init__(self, model='bert-base-cased', special_tokens=False, device='cpu'):
        super(BERT, self).__init__()
        self.dev = device
        self.config = BertConfig.from_pretrained(model, output_hidden_states=True)
        self.sptoks = special_tokens
        self.mod = BertModel.from_pretrained(model, config=self.config).to(self.dev)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.type = 'BERT'

    def E(self, text, level=7, clip_at=500):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))

        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.mod(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))
        outputs = self.mod(torch.LongTensor(ids[steps[0][0]:steps[0][1]]).unsqueeze(0).to(self.dev))[2][level].squeeze(0)

        if len(steps) > 1:
            for step in steps[1:5]:
                outputs = torch.cat([outputs, self.mod(torch.LongTensor(ids[step[0]:step[1]]).unsqueeze(0).to(self.dev))[2][level].squeeze(0)], dim=0)

        return outputs, tokens

    def E_exp(self, text, level=[8,-1], clip_at=500):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))

        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.mod(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))
        outputs = \
        torch.cat(self.mod(torch.LongTensor(ids[steps[0][0]:steps[0][1]]).unsqueeze(0).to(self.dev))[2], dim=0)[
            level].transpose(0, 1)
        
        outputs = outputs.reshape(outputs.shape[0], -1)
        if len(steps) > 1:
            for step in steps[1:5]:
                y_i = torch.cat(self.mod(torch.LongTensor(ids[step[0]:step[1]]).unsqueeze(0).to(self.dev))[2], dim=0)[
                    level].transpose(0, 1)
                outputs = torch.cat([outputs, y_i.reshape(y_i.shape[0], -1)], dim=0)

        return outputs, tokens

    def forward(self, text, level=7, clip_at=500):
        return self.E(text, level, clip_at)
