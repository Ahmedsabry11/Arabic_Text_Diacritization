from charLevelEncoder import CharLevelEncoder 
from pyarabic.araby import tokenize, strip_tashkeel
from utilities import *
from WordFeatureExtraction import WordFeatureExtraction
from torch.utils.data import Dataset, DataLoader
import textProcessing as tp
from AraVec import AraVecEmbbedding
class Embedding_Dataset(Dataset):
    def __init__(self):
        self.data = load_text("dataset/train_preprocessed.txt")
        self.CharEmbedding=CharLevelEncoder(word_embedding_dim=300, char_embedding_dim=5, hidden_dim=10,num_embeddings=38)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sentence = self.data[idx]
        # extract the label
        labels = tp.extract_diacritics_with_previous_letter(sentence)

        # remove the label from the sentence
        sentence = tp.clear_diacritics(sentence)
        tokens = [
                strip_tashkeel(t)+" "
                for t in tokenize(sentence)
            ]
        return tokens,labels


    def collate_fn(self, batch):
          processed_sentences = []
          processed_labels = []
          for sentence, labels in batch:
              processed_sentences.append(sentence)
              processed_labels.append(labels)
          return processed_sentences, processed_labels

    def extract_sentences_word_embedding(self,train_dataloader):
      charEmbeddingVectors=[]
      labels_batches=[]
      wordEmbedding=AraVecEmbbedding()
      for batch_idx, (batch_sentences, batch_labels) in enumerate(train_dataloader):
          print(batch_idx)
          wordEmbeddingVector=wordEmbedding.AraVec_wordEmbedding(batch_sentences)
          charEmbeddingVector=self.CharEmbedding(wordEmbeddingVector)
          charEmbeddingVectors.append(charEmbeddingVector)
          labels_batches.append(batch_labels)
      return charEmbeddingVectors ,labels_batches