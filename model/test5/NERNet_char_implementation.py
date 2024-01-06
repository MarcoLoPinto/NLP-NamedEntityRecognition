# this model parameters and description could change in test4

import torch.nn as nn
import torch
import numpy as np

from TorchCRF import CRF

class NERNet(nn.Module):
    def __init__(self,  hparams,
                        custom_word_embedding_layer = None,
                        custom_char_embedding_layer = None, # not used
                        loss_fn = None,
                        batch_first = True, # ! remember
                        device = 'cpu'):
        """
        Args:
            - hparams: hyperparams to use, such as:
                - n_labels: number of labels
                - use_crf: if the CRF layer must be generated. If not, loss_fn must be provided
                - embedding_word_shape: the shape of the word embedding, which is (vocab_word_size, embedding_word_dim)
                - embedding_char_shape: the shape of the char embedding, which is (vocab_char_size, embedding_char_dim)
                - embedding_word_padding_idx: padding id in the word embedding
                - embedding_char_padding_idx: padding id in the char embedding
                - freeze_word_embedding: if True, the word embedding weights will not change during training
                - freeze_char_embedding: if True, the char embedding weights will not change during training
                - lstm_parameters: a dictionary to create word (and char) LSTM(s) with parameters:
                    - hidden_size: the hidden size of the LSTM
                    - bidirectional: if the LSTM is bidirectional or not
                    - num_layers: number of layers of the LSTM
                    - dropout: dropout value, between 0 and 1 (if num_layers > 1)
                - general_dropout: the value for the dropout layers in the networks, between 0 and 1
            - custom_word_embedding_layer: if not None, it will be used as word embedding layer, otherwise a new embedding will be generated with shape hparams['embedding_word_shape']
            - custom_char_embedding_layer: if not None, it will be used as char embedding layer, otherwise a new embedding will be generated with shape hparams['embedding_char_shape']
            - loss_fn: the loss to be used. Ignored if use_crf is True. It must be used in this format: loss_function(predictions, labels)
            - batch_first: if True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            - device: the device that the model needs to be attached to
        """
        super().__init__()

        self.n_labels = hparams['n_labels']
        self.device = device
        word_embedding_requires_grad = not hparams['freeze_word_embedding']
        char_embedding_requires_grad = not hparams['freeze_char_embedding']
        use_crf = hparams['use_crf']
        
        # 1a. create the word embedding

        if custom_word_embedding_layer is None and hparams['embedding_word_shape'] is None:
            raise RuntimeError("""Either custom_word_embedding_layer or embedding_word_shape must be provided!""")

        if custom_word_embedding_layer is None:
            self.word_embedding = torch.nn.Embedding(hparams['embedding_word_shape'][0],hparams['embedding_word_shape'][1], padding_idx=hparams['embedding_word_padding_idx'])
        else:
            self.word_embedding = custom_word_embedding_layer
        
        embedding_word_dim = self.word_embedding.embedding_dim
        self.word_embedding.weight.requires_grad = word_embedding_requires_grad

        # 1b. create the char embedding

        if custom_char_embedding_layer is None and hparams['embedding_char_shape'] is None:
            raise RuntimeError("""Either custom_char_embedding_layer or embedding_char_shape must be provided!""")

        if custom_char_embedding_layer is None:
            self.char_embedding = torch.nn.Embedding(hparams['embedding_char_shape'][0],hparams['embedding_char_shape'][1], padding_idx=hparams['embedding_char_padding_idx'])
        else:
            self.char_embedding = custom_char_embedding_layer
        
        embedding_char_dim = self.char_embedding.embedding_dim
        self.char_embedding.weight.requires_grad = char_embedding_requires_grad
        
        # 2a. char LSTM

        lstm_char_hidden_size = 64
        lstm_char_bidirectional = True
        lstm_char_num_layers = 3
        lstm_char_dropout = 0.4

        self.lstm_char = nn.LSTM(
            input_size = embedding_char_dim, 
            hidden_size = lstm_char_hidden_size, 

            bidirectional = lstm_char_bidirectional, 
            num_layers = lstm_char_num_layers, 
            dropout = lstm_char_dropout if lstm_char_num_layers > 1 else 0,
            batch_first = batch_first 
        )

        lstm_char_output_dim = 2*lstm_char_hidden_size if lstm_char_bidirectional is True else lstm_char_hidden_size

        # 2b. word + char LSTM

        self.lstm_main = nn.LSTM(
            input_size = embedding_word_dim + lstm_char_output_dim, 
            hidden_size = hparams['lstm_parameters']['hidden_size'], 

            bidirectional = hparams['lstm_parameters']['bidirectional'], 
            num_layers = hparams['lstm_parameters']['num_layers'], 
            dropout = hparams['lstm_parameters']['dropout'] if hparams['lstm_parameters']['num_layers'] > 1 else 0,
            batch_first = batch_first 
        )
        lstm_word_output_dim = 2*hparams['lstm_parameters']['hidden_size'] if hparams['lstm_parameters']['bidirectional'] is True else hparams['lstm_parameters']['hidden_size']

        # 3. classifier (and optionally crf)

        self.classifier = nn.Linear(lstm_word_output_dim, self.n_labels) # TODO

        if use_crf is False and loss_fn is None:
            raise RuntimeError("""Either use_crf = True or loss_fn must be provided!""")
        if use_crf:
            self.crf = CRF(self.n_labels) # , batch_first=batch_first)
        else:
            self.crf = None
            self.loss_fn = loss_fn

        # Extra layers repeated during the net

        self.dropout = nn.Dropout(hparams['general_dropout'])
        self.relu = nn.ReLU()

        self.to(device) # finally, put the model in the right device


    def compute_outputs(self, x_word, x_char):
        ''' compute the model until the classifier layer (before CRF) '''
        
        x_word = self.word_embedding(x_word)
        x_char = self.char_embedding(x_char)

        x_word = self.dropout(x_word)
        x_char = self.dropout(x_char)

        batch_size, sentence_len, word_len, hidden_dim = x_char.size()
        x_char = x_char.view(batch_size * sentence_len, word_len, hidden_dim)
        x_char, _ = self.lstm_char(x_char)
        x_char = x_char[:,-1,:] # take last output of LSTM 
        x_char = x_char.reshape(batch_size, sentence_len, -1)

        x_word = torch.cat((x_word, x_char),dim=-1)

        x_word = self.dropout(x_word)

        x_word, _ = self.lstm_main(x_word)

        x_word = self.classifier(x_word)

        return x_word

    def compute_loss(self, x, y_true, mask = None):
        ''' compute negative log-likelihood (loss_fn() if use_crf = False). Mask is only used when use_crf = True '''
        if self.crf is None:
            return self.loss_fn(x, y_true)

        loss = -(
            torch.sum(self.crf.forward(x,y_true,mask)) / len(x)
        )

        return loss
        
    def forward(self, x_word, x_char, mask = None):
        ''' predict labels '''
        x = self.compute_outputs(x_word, x_char)
        if self.crf is not None:
            x = self.crf.viterbi_decode(x, mask=mask)
        return x

    def get_indices(self, torch_outputs):
        """
        Args:
            torch_outputs (Tensor): a Tensor with shape (batch_size, max_len, label_vocab_size) containing the logits outputed by the neural network (if batch_first = True)
        Output:
            The method returns a (batch_size, max_len) shaped tensor (if batch_first = True)
        """
        max_indices = torch.argmax(torch_outputs, -1) # resulting shape = (batch_size, max_len)
        return max_indices
    
    def load_weights(self, path, strict = True):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=strict)
        self.eval()
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
