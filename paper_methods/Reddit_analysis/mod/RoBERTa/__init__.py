from .RoBERTa import RoBERTa
from .preprocess import pos, sentences

def preprocess(text, wv_mod):
    return pos(text, wv_mod=wv_mod)