from gensim.models import Word2Vec
# CBOW model
class WordFeatureExtraction():

    def __init__(self, tokenized_texts,embedding_dim = 300,window_size = 5,min_count = 1,epoches=5,alpha=0.001):
        self.tokenized_texts=tokenized_texts
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.epoches=epoches
        self.alpha=alpha
    def CBOW(self):
        cbow_model = Word2Vec(sentences=self.tokenized_texts, vector_size=self.embedding_dim, window=self.window_size, sg=0, min_count=self.min_count)
        return cbow_model.wv