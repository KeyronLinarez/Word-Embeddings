import logging
import gensim.models
from gensim import utils
from gensim.test.utils import datapath
import numpy as np    

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('D:/Pyt-hon/Lib/site-packages/gensim/test/test_data/lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
# model.build_vocab(sentences)  # Build the vocabulary
# model.train(sentences, total_examples=model.corpus_count, epochs=10)  

print("Hello")
# gensim will delete any word that doesn't appear more than 5 times
print(model.wv.most_similar(positive=['car', 'vehicle'], topn=5))
vec_king = model.wv['king']
print(vec_king)

model.wv.save_word2vec_format('model.bin', binary=False)

from gensim.models import KeyedVectors
kv = KeyedVectors(512)
kv.add(model.words, model.wv)
kv.save(model.kvmodel)

# trying out smaller toy corpus
sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
           ['i', 'dont', 'drive', 'fast', 'cars'],
           ['data', 'science', 'is', 'fun'],
           ['chocolate', 'is', 'my', 'favorite'],
           ['my', 'favorite', 'movie', 'is', 'predator']]


