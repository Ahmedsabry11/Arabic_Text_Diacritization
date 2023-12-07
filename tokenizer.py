# from tokenizers import ByteLevelBPETokenizer
# import textProcessing as tp


# # Sample Arabic words
# arabic_words = ["مرحبا", "كيف", "حالك", "اليوم"]
# with open("arabic_words.txt", "w", encoding="utf-8") as f:
#     for word in arabic_words:
#         f.write(word + "\n")

# # Create and train a Byte Pair Encoding tokenizer
# tokenizer = ByteLevelBPETokenizer()
# tokenizer.add_tokens(list(tp.ARABIC_LETTERS))
# tokenizer.train(files="dataset/train_preprocessed.txt", vocab_size=10000, min_frequency=5, show_progress=True)

# # Save the trained tokenizer model
# tokenizer.save("arabic_bpe_tokenizer.json")

# # Encode a sample sentence
# encoded_sentence = tokenizer.encode("مرحبا كيف حالك اليوم ")
# print("Encoded Sentence:", encoded_sentence.tokens)
# print("Token IDs:", encoded_sentence.ids)

# decoded_sentence = tokenizer.decode(encoded_sentence.ids)
# print("Decoded Sentence:", decoded_sentence)

import sentencepiece as spm

# Sample sentence
sentence = "قوله"

# Train a SentencePiece model
spm.SentencePieceTrainer.train(input='dataset/undiacritized_train_preprocessed.txt',model_type='bpe', model_prefix='arabic_tokenizer', vocab_size=50000)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='arabic_tokenizer.model')

# Tokenize the sentence
tokens = sp.encode_as_pieces(sentence)
ids = sp.encode_as_ids(sentence)

# Print the results
print("Tokens:", tokens)
print("Token IDs:", ids)
