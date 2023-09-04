import spacy
import torch
import nltk.data

parser = spacy.load('en_core_web_trf')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def sentences(text):
    sents = sent_detector.tokenize(text)
    sents = [sent.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('—', ' ') for sent in sents]
    return sents


def root(text, wv_mod):
    parse = []
    for t in parser(text):
        if 'ROOT' in t.dep_:
            parse += [(t.text, 1)]
        else:
            parse += [(t.text, 0)]

    ids, labels = [], []
    for i, item in enumerate(parse):
        if i > 0:
            k = wv_mod.tokenizer.encode('Ġ' + item[0], add_special_tokens=False)
        else:
            k = wv_mod.tokenizer.encode(item[0], add_special_tokens=False)
        ids += k
        labels += [item[1] for _ in range(len(k))]

    if wv_mod.sptoks:
        sos, eos = wv_mod.tokenizer.special_tokens_map['cls_token'], wv_mod.tokenizer.special_tokens_map['sep_token']
        sos, eos = wv_mod.tokenizer.convert_tokens_to_ids(sos), wv_mod.tokenizer.convert_tokens_to_ids(eos)
        ids = [sos] + ids + [eos]
        labels = [0] + labels + [0]

    return torch.LongTensor(ids), torch.FloatTensor(labels), len(parse)

def dependency(text,wv_mod):
    parse = []
    for t in parser(text):
        if 'nsubj' in t.dep_:
            parse += [(t.text, 1)]
        elif 'obj' in t.dep_:
            parse += [(t.text, 1)]
        elif 'neg' in t.dep_:
            parse += [(t.text, 1)]
        elif 'ROOT' in t.dep_:
            parse += [(t.text, 1)]
        elif t.pos_ in ['VERB', 'INTJ']:
            parse += [(t.text, 1)]
        else:
            parse += [(t.text, 0)]

    ids, labels = [], []
    for i,item in enumerate(parse):
        if i > 0:
            k = wv_mod.tokenizer.encode('Ġ'+item[0], add_special_tokens=False)
        else:
            k = wv_mod.tokenizer.encode(item[0], add_special_tokens=False)
        ids+=k
        labels+=[item[1] for _ in range(len(k))]

    if wv_mod.sptoks:
        sos, eos = wv_mod.tokenizer.special_tokens_map['cls_token'], wv_mod.tokenizer.special_tokens_map['sep_token']
        sos, eos = wv_mod.tokenizer.convert_tokens_to_ids(sos), wv_mod.tokenizer.convert_tokens_to_ids(eos)
        ids = [sos] + ids + [eos]
        labels = [0] + labels + [0]

    return torch.LongTensor(ids), torch.FloatTensor(labels), len(parse)

def dependency_(text,wv_mod):
    parse = []
    for t in parser(text):
        if 'nsubj' in t.dep_:
            parse += [(t.text, 1)]
        elif 'obj' in t.dep_:
            parse += [(t.text, 1)]
        elif 'neg' in t.dep_:
            parse += [(t.text, 1)]
        elif 'root' in t.dep_:
            parse += [(t.text, 1)]
        elif t.pos_ in ['VERB', 'INTJ']:
            parse += [(t.text, 1)]
        else:
            parse += [(t.text, 0)]

    ids, labels = [], []
    for i, item in enumerate(parse):
        if i > 0:
            k = wv_mod.tokenizer.encode('Ġ' + item[0], add_special_tokens=False)
        else:
            k = wv_mod.tokenizer.encode(item[0], add_special_tokens=False)
        ids += k
        labels += [item[1]] + [0 for _ in range(len(k)-1)]

    if wv_mod.sptoks:
        sos, eos = wv_mod.tokenizer.special_tokens_map['cls_token'], wv_mod.tokenizer.special_tokens_map['sep_token']
        sos, eos = wv_mod.tokenizer.convert_tokens_to_ids(sos), wv_mod.tokenizer.convert_tokens_to_ids(eos)
        ids = [sos] + ids + [eos]
        labels = [0] + labels + [0]

    return torch.LongTensor(ids), torch.FloatTensor(labels), len(parse)

def pos(text, wv_mod):
    parse = []
    for t in parser(text):
        if t.pos_ in ['NOUN', 'ADJ', 'PRON', 'VERB', 'ADV', 'AUX', 'INTJ']:
            parse += [(t.text, 1)]
        elif ('neg' in t.dep_) or ('ROOT' in t.dep_):
            parse += [(t.text, 1)]
        else:
            parse +=[(t.text, 0)]

    ids, labels = [], []
    for i, item in enumerate(parse):
        if i > 0:
            k = wv_mod.tokenizer.encode('Ġ' + item[0], add_special_tokens=False)
        else:
            k = wv_mod.tokenizer.encode(item[0], add_special_tokens=False)
        ids += k
        labels += [item[1] for _ in range(len(k))]

    if wv_mod.sptoks:
        sos, eos = wv_mod.tokenizer.special_tokens_map['cls_token'], wv_mod.tokenizer.special_tokens_map['sep_token']
        sos, eos = wv_mod.tokenizer.convert_tokens_to_ids(sos), wv_mod.tokenizer.convert_tokens_to_ids(eos)
        ids = [sos] + ids + [eos]
        labels = [0] + labels + [0]

    return torch.LongTensor(ids), torch.FloatTensor(labels), len(parse)