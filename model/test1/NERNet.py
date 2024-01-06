
import torch.nn as nn
import torch
import numpy as np

from NERDataset import NERDataset, PAD_TOKEN, window_size, window_shift

class NERNet(nn.Module):
    def __init__(self, gensim_embedding, output_size, hidden_size = 128, device = 'cpu'):
        super(NERNet, self).__init__()
        
        bidirectional = True
        lstm1_layers = 2
        lstm1_dropout = 0.2 if lstm1_layers > 1 else 0
        hidden_size_classifier = hidden_size*2 if bidirectional else hidden_size
        self.device = device
        
        self.embedding = gensim_embedding.wv
        
        self.seq_encoder = nn.LSTM(
            input_size=gensim_embedding.vector_size, 
            hidden_size=hidden_size, 
            bidirectional=bidirectional, 
            num_layers=lstm1_layers, 
            dropout=lstm1_dropout
        ) 
        # self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=0.48)
        # self.dropout = nn.Dropout(0.4)
        
        self.classifier = nn.Linear(hidden_size_classifier, output_size)
        
    def forward(self, x):
        for i in range(len(x)):
            x[i] = self.embedding[x[i]]
        x = torch.tensor(x).to(self.device)
        x, (h, c) = self.seq_encoder(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x
        
    def predict(self, tokens):
        """
        Args:
            tokens: list of list of strings. The outer list represents the sentences, the inner one the tokens contained
            within it. Ex: [ ["This", "is", "the", "first", "homework"], ["Barack", "Obama", "was", "elected"] ]
        Returns:
            list of list of predictions associated to each token in the respective position.
            Ex: Ex: [ ["O", "O", "O", "O", "O"], ["PER", "PER", "O", "O"] ]
        """
        self.eval() # dropout to 0
        self.to(self.device)

        predictions = []

        with torch.no_grad():
            for sentence in tokens:
                windowed_sentence = NERDataset.generate_windows([sentence], window_size, window_size, pad_element = PAD_TOKEN)
                inputs = np.array(windowed_sentence)
                inputs = inputs.transpose( (1,0) ).tolist()
                y_pred = self.get_indices( self(inputs) ).transpose(0,1)
                y_pred = y_pred.reshape(-1).cpu().numpy().tolist()[:len(sentence)]
                predictions.append( y_pred )
                
        return predictions

    def get_indices(self, torch_outputs):
        """
        Args:
            torch_outputs (Tensor): a Tensor with shape (batch_size, max_len, label_vocab_size) or (max_len, batch_size, label_vocab_size) containing the logits outputed by the neural network.
        Output:
            The method returns a tensor.
        """
        max_indices = torch.argmax(torch_outputs, -1) # resulting shape = (batch_size, max_len)
        return max_indices
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
