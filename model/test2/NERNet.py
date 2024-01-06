# this model parameters and description could change in test3

import torch.nn as nn
import torch
import numpy as np

class NERNet(nn.Module):
    def __init__(self, hidden_size = 80, output_size = 13, custom_embedding_layer = None, vocab_dim = None, embedding_dim = None, device = 'cpu'):
        """
        Args:
            - hidden_size: the hidden size of the model in the sequence encoder layer (LSTM)
            - output_size: number of labels
            - custom_embedding_layer: if not None, it will be used as embedding layer, otherwise a new embedding will be generated with shape (vocab_dim, embedding_dim)
            - device: the device that the model needs to be attached to
        """
        super(NERNet, self).__init__()
        
        bidirectional = True
        lstm1_layers = 2
        lstm1_dropout = 0.2 if lstm1_layers > 1 else 0
        hidden_size_classifier = hidden_size*2 if bidirectional else hidden_size
        self.device = device
        
        if custom_embedding_layer is None:
            self.embedding = torch.nn.Embedding(vocab_dim, embedding_dim)
        else:
            self.embedding = custom_embedding_layer
            embedding_dim = custom_embedding_layer.embedding_dim
        
        self.seq_encoder = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            bidirectional=bidirectional, 
            num_layers=lstm1_layers, 
            dropout=lstm1_dropout,
            batch_first=True # ! remember
        ) 
        
        self.classifier = nn.Linear(hidden_size_classifier, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.seq_encoder(x)
        x = self.classifier(x)
        return x

    def get_indices(self, torch_outputs):
        """
        Args:
            torch_outputs (Tensor): a Tensor with shape (batch_size, max_len, label_vocab_size) containing the logits outputed by the neural network.
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
