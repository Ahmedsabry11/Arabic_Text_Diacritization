import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer

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

create_tokenized_sentence2("dataset/undiacritized_train_preprocessed.txt","arabic_bpe_tokenizer.json")
# create_tokenized_sentence("dataset/undiacritized_train_preprocessed.txt","arabic_tokenizer.model")
    
