import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer
import textProcessing as tp
from pyarabic.araby import tokenize, strip_tashkeel


class Tokenizer:
    def __init__(self,tokenizer_type, model_file="",load = True):
        self.model_file = model_file
        self.tokenizer = None
        self.tokenizer_type = tokenizer_type
        if load:
            self.load_tokenizer()
        

    def train_tokenizer(self, file_path, model_type, model_prefix, vocab_size):
        if self.tokenizer_type == "sentencepiece":
            spm.SentencePieceTrainer.train(input=file_path,model_type=model_type, model_prefix=model_prefix, vocab_size=vocab_size)
            self.model_file = model_prefix + ".model"
        elif self.tokenizer_type == "bytelevelbpe":
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=file_path, vocab_size=vocab_size, min_frequency=2, show_progress=True)
            tokenizer.save(model_prefix)
            tokenizer.save_model("./")
        elif self.tokenizer_type == "pyarabic":
            # use pyarabic tokenizer
            pass
        else:
            print("Error: Invalid tokenizer type")
            exit()

    def load_tokenizer(self):
        if self.tokenizer_type == "sentencepiece":
            self.tokenizer = self.load_sentencepiece_tokenizer()
        elif self.tokenizer_type == "bytelevelbpe":
            self.tokenizer = self.load_bytelevelbpe_tokenizer()
        elif self.tokenizer_type == "pyarabic":
            # use pyarabic tokenizer
            pass
        else:
            print("Error: Invalid tokenizer file")
            exit()

    def load_sentencepiece_tokenizer(self):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=self.model_file)
        return sp
    
    def load_bytelevelbpe_tokenizer(self):
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer.from_file(merges_filename = "merges.txt",vocab_filename = "vocab.json")
        return tokenizer
    
    def encode(self, sentence):
        if self.tokenizer_type == "sentencepiece":
            return self.tokenizer.encode_as_pieces(sentence)
        elif self.tokenizer_type == "bytelevelbpe":
            encoded_sentence = self.tokenizer.encode(sentence)
            sentence_words = []
            for id in encoded_sentence.ids:
                decoded_word = self.tokenizer.decode([id])
                # append only words
                sentence_words.append(decoded_word)
            return sentence_words
        elif self.tokenizer_type == "pyarabic":
            # use pyarabic tokenizer
            # tokenize the sentence
            tokens = tokenize(sentence)
            # remove tashkeel
            # tokens = strip_tashkeel(tokens)
            return tokens

        else:
            print("Error: Invalid tokenizer type")
            exit()
            
    def decode(self, sentence):
        if self.tokenizer_type == "sentencepiece":
            return self.tokenizer.decode(sentence)
        elif self.tokenizer_type == "bytelevelbpe":
            decoded_sentence = ""
            for id in sentence:
                decoded_sentence += self.tokenizer.decode([id])
            return decoded_sentence
        elif self.tokenizer_type == "pyarabic":
            # decode using pyarabic
            decoded_sentence = ""
            for word in sentence:
                decoded_sentence += word
            return decoded_sentence
        else:
            print("Error: Invalid tokenizer type")
            exit()
            
    def tokenize(self, sentence):
        if self.tokenizer_type == "sentencepiece":
            return self.tokenizer.encode_as_pieces(sentence)
        elif self.tokenizer_type == "bytelevelbpe":
            encoded_sentence = self.tokenizer.encode(sentence)
            sentence_words = []
            for id in encoded_sentence.ids:
                decoded_word = self.tokenizer.decode([id])
                # append only words
                sentence_words.append(decoded_word)
            return sentence_words
        elif self.tokenizer_type == "pyarabic":
            # use pyarabic tokenizer
            # tokenize the sentence
            tokens = tokenize(sentence)
            # remove tashkeel
            # tokens = strip_tashkeel(tokens)
            return tokens
        else:
            print("Error: Invalid tokenizer type")
            exit()

    def tokenize_sentences(self, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentences.append(self.tokenize(sentence))
        return tokenized_sentences
    
# from tokenizers import ByteLevelBPETokenizer
# import textProcessing as tp

# # Create and train a Byte Pair Encoding tokenizer
# tokenizer = ByteLevelBPETokenizer()
# # tokenizer.add_tokens()
# tokenizer.train(files="dataset/undiacritized_train_preprocessed.txt", vocab_size=50000, min_frequency=2, show_progress=True)

# # Save the trained tokenizer model
# tokenizer.save("arabic_bpe_tokenizer.json")
# tokenizer.save_model("./")
# # tokenizer.save()
# # Encode a sample sentence
# encoded_sentence = tokenizer.encode(" قطع الأول يده إلخ  قال الزركشي")
# print("Encoded Sentence:", encoded_sentence.tokens)

# print("Token IDs:", encoded_sentence.ids)

# decoded_sentence = tokenizer.decode(encoded_sentence.ids)
# print("Decoded Sentence:", decoded_sentence)
# print("Decoded Sentence:", type(decoded_sentence))

# # decode id by id
# decoded_sentence = ""
# for id in encoded_sentence.ids:
#     decoded_sentence = tokenizer.decode([id])
#     print("Decoded Sentence:", decoded_sentence)


# import sentencepiece as spm

# # Sample sentence
# sentence = "قوله"

# # Train a SentencePiece model
# spm.SentencePieceTrainer.train(input='dataset/undiacritized_train_preprocessed.txt',model_type='bpe', model_prefix='arabic_tokenizer', vocab_size=50000)

# # Load the trained SentencePiece model
# sp = spm.SentencePieceProcessor(model_file='arabic_tokenizer.model')

# # Tokenize the sentence
# tokens = sp.encode_as_pieces(sentence)
# ids = sp.encode_as_ids(sentence)

# # Print the results
# print("Tokens:", tokens)
# print("Token IDs:", ids)
