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
        super(CharLevelEncoder, self).__init__()
        # Arabic characters
        arabic_characters =  frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
        

        # Create a dictionary to map characters to indices
        self.char_to_index =dict((l, n) for n, l in enumerate(sorted(arabic_characters)))
        self.char_to_index.update(dict((v, k) for k, v in enumerate([' ', '0'], len(self.char_to_index))))


        # Create a dictionary to map indices to characters
        self.index_to_char = {index: char for index, char in enumerate(arabic_characters)}

        self.word_embedding_dim = word_embedding_dim

        # Character-level embeddings
        self.char_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=char_embedding_dim)  # Example: ASCII character range

        # LSTM for character-level information
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # Linear layer for combining word and character embeddings
        self.linear = nn.Linear(word_embedding_dim + hidden_dim, word_embedding_dim)  # Adjust output dimensions

    def forward(self, word_embeddings):
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

                        char_indices = [self.char_to_index[char] for char in arabic_word_chars]


                         # Convert word embedding to tensor
                        word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float32)
                        word_embedding_tensor = word_embedding_tensor.view(1,-1)
                        word_embedding_tensor = word_embedding_tensor.expand(len(arabic_word_chars),-1)


                        # Get character-level embeddings

                        char_embedded = self.char_embedding(torch.tensor(char_indices))

                        # Reshape char_embedded for LSTM input
                        char_embedded = char_embedded.view(len(arabic_word_chars), 1,-1)

                        # Get character-level LSTM output
                        _, (hidden, _) = self.char_lstm(char_embedded)

                        # Concatenate word and character embeddings
                        # print( word_embedding_tensor.shape,"hell")


                        combined = torch.cat((word_embedding_tensor, hidden.squeeze(0)), dim=1)
                        # Apply linear layer to combine embeddings
                        combined = F.relu(self.linear(combined))
                        combined_reshaped = combined.view(-1, combined.size(-1))
                        char_embedding_dict = {}
                        for idx, char_idx in enumerate(char_indices):
                            char_embedding_dict[arabic_word_chars[idx]] = combined_reshaped[idx].tolist()
                        sentence_char_list.append(char_embedding_dict)

                    all_sentence_list.append(sentence_char_list)

        return sentence_char_list