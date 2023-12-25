import gensim
from gensim.models import KeyedVectors
from gensim.models import word2vec
class AraVecEmbbedding():
    def __init__(self,embedding_dim = 300,window_size = 5,min_count = 1,epoches=5,alpha=0.001):
        self.AraVec_model= KeyedVectors.load("tweets_cbow_300")


    def map_words_to_vectors(self,input_list, word_vector_model):
        output = []
        for inner_list in input_list:
            inner_output = {}
            for word in inner_list:
                if word in word_vector_model.wv.index_to_key:
                    inner_output[word] = word_vector_model.wv.get_vector(word)
            if inner_output:
                output.append(inner_output)
        return output

    # def AraVec_load(self):
      # self.AraVec_model= KeyedVectors.load("tweets_cbow_300")
  
    def AraVec_wordEmbedding(self,tokenized_texts):
        return self.map_words_to_vectors(tokenized_texts,self.AraVec_model)
