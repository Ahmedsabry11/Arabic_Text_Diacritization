import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer
import textProcessing as tp
import numpy as np

def load_text(file_path):
    # read text file sentence by sentence
    sentences = []
    with open(file_path,'rt', encoding='utf-8') as f:
        text_line = f.readline().strip()
        # train_text.append(train_text_line)
        while(text_line != ""):
            sentences.append(text_line)
            text_line = f.readline().strip()
    return sentences

def extract_labels(sentences):
    labels = []
    for sentence in sentences:
        # extract diacritics
        diacritics = tp.extract_diacritics_with_previous_letter(sentence)
        # append diacritics to labels
        labels.append(diacritics)
    return labels


def create_tokenized_sentence(file_path,model_file):
    # read text file sentence by sentence
    sentences = []
    with open(file_path,'rt', encoding='utf-8') as f:
        text_line = f.readline().strip()
        # train_text.append(train_text_line)
        while(text_line != ""):
            sentences.append(text_line)
            text_line = f.readline().strip()
    # apply encoding using sentencepiece
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=model_file)
    # Encode
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentences.append(sp.encode_as_pieces(sentence))
    # print first 10 sentences
    print(encoded_sentences[:10])
    return encoded_sentences

def create_tokenized_sentence2(file_path,model_file):
    # read text file sentence by sentence
    sentences = []
    with open(file_path,'rt', encoding='utf-8') as f:
        text_line = f.readline().strip()
        # train_text.append(train_text_line)
        while(text_line != ""):
            sentences.append(text_line)
            text_line = f.readline().strip()
    # apply encoding using sentencepiece
    # Load the trained tokenizer model
    tokenizer = ByteLevelBPETokenizer.from_file(merges_filename = "merges.txt",vocab_filename = "vocab.json")
    # Encode
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentence = tokenizer.encode(sentence)
        # print("Encoded Sentence:", encoded_sentence.tokens)
        sentence_words = []
        for id in encoded_sentence.ids:
            decoded_word = tokenizer.decode([id])
            # print("Decoded Sentence:", decoded_word)
            # append only words
            sentence_words.append(decoded_word)
        encoded_sentences.append(sentence_words)
            
        
    # print first 10 sentences
    print(encoded_sentences[:10])
    # print(sentences[:10])
    return encoded_sentences

"""
    input is 2d array of sentences

    each sentence is an array of words

    each word is an array of characters

    each character is vector of 1s and 0s or embedding with dim 300

    get_item should return a sentence as a 2d array of characters

"""
def convert_char_to_vector(char):
    # convert each character to vector of 1s and 0s
    # get character index from dictionary
    index = tp.CHAR2INDEX[char]
    # create vector of zeros
    vector = [0] * len(tp.CHAR2INDEX)
    # set index to 1
    vector[index] = 1
    return np.array(vector)

def convert_diacritic_to_vector(diacritic):
    # convert each diacritic to vector of 1s and 0s
    # get diacritic index from dictionary

    index = tp.DIACRITIC2INDEX[diacritic]
    # create vector of zeros
    vector = [0] * len(tp.DIACRITIC2INDEX)
    # set index to 1
    vector[index] = 1
    return np.array(vector)

def convert_sentence_to_vector(tokenzied_sentence):
    # convert each character to vector of 1s and 0s
    # get character index from dictionary
    sentence_vector = []
    for word in tokenzied_sentence:
        for char in word:
            char_vec = convert_char_to_vector(char)
            sentence_vector.append(char_vec)
    return np.array(sentence_vector)

# create_tokenized_sentence2("dataset/undiacritized_train_preprocessed.txt","arabic_bpe_tokenizer.json")
# create_tokenized_sentence("dataset/undiacritized_train_preprocessed.txt","arabic_tokenizer.model")
    
