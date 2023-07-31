import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        #2. Initialize RNN layer
        #3. Initialize a fully connected layer with Linear transformation
        #4. Initialize Dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers = n_layers, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        #2. Run the RNN along the sentences of length sent_len. #output = [sent len, batch size, hid dim * num directions]; #hidden = [num layers * num directions, batch size, hid dim]
        #3. Get last forward (hidden[-1,:,:]) hidden layer and apply dropout
        #text = text.permute(1,0)
        embed = self.embedding(text)
        embed_drop = self.dropout(embed)
        # packed = nn.utils.rnn.pack_padded_sequence(embed_drop, text_lengths, 
        #                                            batch_first=False,
        #                                            enforce_sorted=False)
        o, h = self.rnn(embed_drop)
        h = h[-1,:,:];
        h = self.dropout(h);    
        return self.fc(h)