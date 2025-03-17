## In this script we will create modules for the doc2vec/word2vec training

import gensim

def doc2vec_dm(vector_size = 100): 
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=5, epochs=60, workers = 7, dm = 0, 
                                          dbow_words = 0, window = 7)
    return model


def doc2vec_dbow(vector_size = 100): 
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=5, epochs=60, workers = 7, dm = 1, window = 7)
    return model


def word2vec_cbow(vector_size = 100): 
    model = gensim.models.Word2Vec(vector_size=vector_size, window = 7, min_count = 5, max_vocab_size = None, sg = 0, workers = 6, epochs=60)
    return model    


def word2vec_sg(vector_size = 100): 
    model = gensim.models.Word2Vec(vector_size=vector_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1, workers = 6, epochs=60)
    return model   


def fasttext_sg(vector_size = 100): 
    model = gensim.models.fasttext.FastText(vector_size=vector_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1, workers = 6, epochs=60)
    return model   


def fasttext_cbow(vector_size = 100): 
    model = gensim.models.fasttext.FastText(vector_size=vector_size, window = 7, min_count = 5, max_vocab_size = None, sg = 0, workers = 6, epochs=60)
    return model   