import torch
import pandas as pd
import numpy as np

import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig



class RoBERTa(nn.Module):

    def __init__(self, model='roberta-base', special_tokens=False, device='cpu', layers=[7,-1], max_sequence_length=500, clip_at=400, pre_sequence_stride=100):
        super(RoBERTa, self).__init__()
        self.dev = device                   # whether running on 'cuda' or 'cpu'

        # BERT Model components
        self.config = RobertaConfig.from_pretrained(model, output_hidden_states=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.mod = RobertaModel.from_pretrained(model, config=self.config).to(self.dev)

        # Vector manipulations
        self.layers = layers                # which hidden layers to attenuate to
        self.flat = nn.Flatten(-2,-1)       # protocol for flattening attenuated hidden layers

        # Sequence manipulations
        self.max_windows = 5                # the maximum number of windows for overflow tokens to take
        self.sptoks = special_tokens        # whether or not to return special tokens
        self.max_seq = max_sequence_length  # maximum unclipped sequence length
        self.stride = pre_sequence_stride   # how many tokens to include prior to sequence when clipping needed
        self.clip = clip_at                 # window length in tokens (excluding pre_sequence_stride)

    def _tokenize(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))
        return ids, tokens

    def _vectorize(self, ids):

        outputs = torch.cat(
            self.mod(
                torch.LongTensor(ids).unsqueeze(0).to(self.dev)
            )[2],
            dim=0
        )[self.layers].transpose(0, 1)

        return self.flat(outputs)

    def E(self, ids):

        if len(ids) <= self.max_seq:
            outputs = self._vectorize(ids)

        else:

            nSpans = int(len(ids) / self.clip)
            start = [i * self.clip for i in range(nSpans)] + [nSpans * self.clip]
            fins = [(i + 1) * self.clip for i in range(nSpans)] + [len(ids)]
            steps = list(zip(start, fins))

            outputs = self._vectorize(ids[steps[0][0]:steps[0][1]])

            for step in steps[1:self.max_windows]:
                y_i = self._vectorize(ids[step[0]-self.stride:step[1]])[self.stride:]
                outputs = torch.cat([outputs, y_i], dim=0)

        return outputs

    def forward(self, text, prompt=None, level=None, clip_cls=True):

        if level != None:
            self.layers=level

        ids, tokens = self._tokenize(text)
        delta = None

        if prompt != None:

            _ids,_tokens = self._tokenize(prompt)
            ids, tokens = _ids+ids, np.concatenate([_tokens, tokens])
            delta = len(_ids)

        Ex = self.E(ids)

        return Ex, tokens, delta



class BERT_old(nn.Module):

    def __init__(self, model='bert-base-cased', special_tokens=False, device='cpu'):
        super(BERT_old, self).__init__()
        self.dev = device
        self.config = BertConfig.from_pretrained(model, output_hidden_states=True)
        self.sptoks = special_tokens
        self.mod = BertModel.from_pretrained(model, config=self.config).to(self.dev)
        self.tokenizer = BertTokenizer.from_pretrained(model)

    def tokenize(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))
        return ids, tokens

    def E(self, ids, level=[8,-1], clip_at=500):
        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        steps = list(zip(start, fins))
        outputs = torch.cat(self.mod(torch.LongTensor(ids[steps[0][0]:steps[0][1]]).unsqueeze(0).to(self.dev))[2], dim=0)[level].transpose(0,1)
        outputs = outputs.reshape(outputs.shape[0],-1)

        if len(steps) > 1:
            for step in steps[1:5]:
                y_i = torch.cat(self.mod(torch.LongTensor(ids[step[0]:step[1]]).unsqueeze(0).to(self.dev))[2], dim=0)[level].transpose(0,1)
                outputs = torch.cat([outputs, y_i.reshape(y_i.shape[0],-1)], dim=0)

        return outputs

    def forward(self, text, prompt=None, level=[7,-1], clip_at=500):
        ids, tokens = self.tokenize(text)
        delta = None

        if prompt != None:
            _ids,_tokens = self.tokenize(prompt)

            if self.sptoks:
                _ids, _tokens = _ids[:-1],_tokens[:-1]
                ids,tokens = ids[1:],tokens[1:]

            ids, tokens = _ids+ids, np.concatenate([_tokens, tokens])
            delta = len(_ids)

        return self.E(ids, level, clip_at), tokens, delta



class __BERT__(nn.Module):

    def __init__(self, model='bert-base-cased', special_tokens=False, device='cpu'):
        super(__BERT__, self).__init__()
        self.dev = device
        self.config = BertConfig.from_pretrained(model, output_hidden_states=True)
        self.sptoks = special_tokens
        self.mod = BertModel.from_pretrained(model, config=self.config).to(self.dev)
        self.tokenizer = BertTokenizer.from_pretrained(model)

    def E(self, text, level=7, clip_at=500):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))

        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.generator(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
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

        # outputs = [self.generator(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))
        outputs = torch.cat(self.mod(torch.LongTensor(ids[steps[0][0]:steps[0][1]]).unsqueeze(0).to(self.dev))[2], dim=0)[level].transpose(0,1)
        outputs = outputs.reshape(outputs.shape[0],-1)
        # outputs = outputs.sum(dim=1)
        if len(steps) > 1:
            for step in steps[1:5]:
                y_i = torch.cat(self.mod(torch.LongTensor(ids[step[0]:step[1]]).unsqueeze(0).to(self.dev))[2], dim=0)[level].transpose(0,1)
                outputs = torch.cat([outputs, y_i.reshape(y_i.shape[0],-1)], dim=0)
                # outputs = torch.cat([outputs, y_i.sum(dim=1)], dim=0)

        return outputs, tokens

    def forward(self, text, level=[7,-1], clip_at=500):
        return self.E_exp(text, level, clip_at)

    def tokenize(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))
        return ids, tokens

    def __from_ids__(self, ids, clip_at=500, level=7):
        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.generator(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))
        outputs = self.mod(ids[steps[0][0]:steps[0][1]].to(self.dev))[2][level].squeeze(0)

        if len(steps) > 1:
            for step in steps[1:5]:
                outputs = torch.cat([outputs,
                                     self.mod(ids[step[0]:step[1]].to(self.dev))[2][level].squeeze(0)
                                     ], dim=0)

        return outputs

    def from_ids(self, ids, clip_at=500, level=[7,-1]):
        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.generator(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))

        outputs = \
        torch.cat(self.mod(ids[steps[0][0]:steps[0][1]].view(1,-1).to(self.dev))[2], dim=0)[
            level].transpose(0, 1)

        outputs = outputs.reshape(outputs.shape[0], -1)
        # outputs = outputs.sum(dim=1)

        if len(steps) > 1:
            for step in steps[1:5]:
                y_i = torch.cat(self.mod(ids[step[0]:step[1]].view(1,-1).to(self.dev))[2], dim=0)[
                    level].transpose(0, 1)
                outputs = torch.cat([outputs, y_i.reshape(y_i.shape[0], -1)], dim=0)
                # outputs = torch.cat([outputs, y_i.sum(dim=1)], dim=0)

        return outputs

    def save_to_file(self, texts, f, level=7, clip_at=500):
        df = pd.DataFrame(columns=['sentN', 'token', 'vec'])
        df.to_csv(f,index=False, encoding='utf-8')
        for i,text in enumerate(texts):
            vecs, tokens = self.E(text, level=level, clip_at=clip_at)
            for tok, vec in list(zip(tokens, vecs)):
                d = pd.DataFrame(np.array([i,tok,str(vec.view(-1).tolist()).replace(', ', ' ').replace('[', '').replace(']', '')]).reshape(1,-1), columns=list(df))
                d.to_csv(f,index=False,header=False,encoding='utf-8',mode='a')

