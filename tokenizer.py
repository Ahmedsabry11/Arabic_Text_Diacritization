from tokenizers import ByteLevelBPETokenizer
import textProcessing as tp

# Create and train a Byte Pair Encoding tokenizer
tokenizer = ByteLevelBPETokenizer()
# tokenizer.add_tokens()
tokenizer.train(files="dataset/undiacritized_train_preprocessed.txt", vocab_size=50000, min_frequency=2, show_progress=True)

# Save the trained tokenizer model
tokenizer.save("arabic_bpe_tokenizer.json")

# Encode a sample sentence
encoded_sentence = tokenizer.encode(" قطع الأول يده إلخ  قال الزركشي")
print("Encoded Sentence:", encoded_sentence.tokens)

print("Token IDs:", encoded_sentence.ids)

decoded_sentence = tokenizer.decode(encoded_sentence.ids)
print("Decoded Sentence:", decoded_sentence)
print("Decoded Sentence:", type(decoded_sentence))

# decode id by id
decoded_sentence = ""
for id in encoded_sentence.ids:
    decoded_sentence = tokenizer.decode([id])
    print("Decoded Sentence:", decoded_sentence)


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
