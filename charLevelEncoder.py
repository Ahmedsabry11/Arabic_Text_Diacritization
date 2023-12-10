import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLevelEncoder(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim,num_embeddings):
        """
        Initialize the CharLevelEncoder module.

        Args:
        - word_embedding_dim (int): Dimensionality of word embeddings.
        - char_embedding_dim (int): Dimensionality of character embeddings.
        - hidden_dim (int): Dimensionality of the hidden state in the LSTM.
        - num_embeddings (int): Number of unique characters to learn embeddings for.
        """
        super(CharEncoder, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        
        # Character-level embeddings
        self.char_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=char_embedding_dim)  # Example: ASCII character range
        
        # LSTM for character-level information
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Linear layer for combining word and character embeddings
        self.linear = nn.Linear(word_embedding_dim + hidden_dim, word_embedding_dim)  # Adjust output dimensions
        
    def forward(self, word_embedding):
        """
        Forward pass of the CharLevelEncoder to generate character-level embeddings for words.

        Args:
        - word_embeddings (list): List of dictionaries containing word embeddings.

        Returns:
        - char_embeddings_list (list): List of sentences, each containing character embeddings for words.
        """

        all_sentence_list = []
        for word_dict in word_embeddings:
                sentence_char_list = []
                for word, word_embedding in word_dict.items():
                        arabic_word_chars = list(word)

                        char_indices = [char.encode('utf-8') for char in arabic_word_chars]

                         # Convert word embedding to tensor
                        word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float32)

                        # Get character-level embeddings

                        char_embedded = self.char_embedding(torch.tensor(char_indices))

                        # Reshape char_embedded for LSTM input
                        #(batch_size,sequence_length,input_size)
                        char_embedded = char_embedded.view(1, -1, char_embedded.size(-1))

                        # Get character-level LSTM output
                        _, (hidden, _) = self.char_lstm(char_embedded)

                        # Concatenate word and character embeddings
                        combined = torch.cat((word_embedding_tensor, hidden.squeeze(0)), dim=1)
                        # Apply linear layer to combine embeddings
                        combined = F.relu(self.linear(combined))
                        combined_reshaped = combined.view(-1, combined.size(-1))

                        char_embedding_dict = {}
                        for idx, char_idx in enumerate(char_indices):
                            char_embedding_dict[arabic_word_chars[idx]] = combined_reshaped[idx].tolist()
                        sentence_char_list.append(char_embedding_dict)

                all_sentence_list.append(sentence_char_list)
                
        return all_sentence_list

#Calling
# char_encoder = CharEncoder(word_embedding_dim=3, char_embedding_dim=5, hidden_dim=10,num_embeddings=128)
# output_dict = char_encoder(word_embeddings)

# print(output_dict)

#Example for the output:
#input: word_embeddings = [
#     {'ahmed': [1, 2, 3], 'went': [3, 4, 5], 'club': [5, 6, 7]},
#     {'ali': [1, 2, 6], 'go': [1, 7, 9]}
# ]
# output: [
#     [  # Sentence 1
#         {'a': [char_embedding_1], 'h': [char_embedding_2], 'm': [char_embedding_3], 'e': [char_embedding_4], 'd': [char_embedding_5]},
#         {'w': [char_embedding_6], 'e': [char_embedding_7], 'n': [char_embedding_8], 't': [char_embedding_9]},
#         {'c': [char_embedding_10], 'l': [char_embedding_11], 'u': [char_embedding_12], 'b': [char_embedding_13]}
#     ],
#     [  # Sentence 2
#         {'a': [char_embedding_14], 'l': [char_embedding_15], 'i': [char_embedding_16]},
#         {'g': [char_embedding_17], 'o': [char_embedding_18]}
#     ]
# ]


