# this model parameters and description could change in test4

import torch.nn as nn
import torch
import numpy as np

from TorchCRF import CRF

class NERNet(nn.Module):
    def __init__(self,  hparams,
                        custom_word_embedding_layer = None,
                        custom_pos_embedding_layer = None, # not used
                        loss_fn = None,
                        batch_first = True, # ! remember
                        device = 'cpu'):
        """
        Args:
            - hparams: hyperparams to use, such as:
                - n_labels: number of labels
                - use_crf: if the CRF layer must be generated. If not, loss_fn must be provided
                - embedding_word_shape: the shape of the word embedding, which is (vocab_word_size, embedding_word_dim)
                - embedding_word_padding_idx: padding id in the word embedding
                - freeze_word_embedding: if True, the word embedding weights will not change during training
                - lstm_parameters: a dictionary to create word LSTM with parameters:
                    - hidden_size: the hidden size of the LSTM
                    - bidirectional: if the LSTM is bidirectional or not
                    - num_layers: number of layers of the LSTM
                    - dropout: dropout value, between 0 and 1 (if num_layers > 1)
                - general_dropout: the value for the dropout layers in the networks, between 0 and 1
            - custom_word_embedding_layer: if not None, it will be used as word embedding layer, otherwise a new embedding will be generated with shape hparams.embedding_word_shape
            - loss_fn: the loss to be used. Ignored if use_crf is True. It must be used in this format: loss_function(predictions, labels)
            - batch_first: if True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            - device: the device that the model needs to be attached to
        """
        super().__init__()

        self.n_labels = hparams['n_labels']
        self.n_pos = hparams['embedding_pos_shape'][0]
        self.device = device
        word_embedding_requires_grad = not hparams['freeze_word_embedding']
        use_crf = hparams['use_crf']
        
        # 1. create the word encoder

        if custom_word_embedding_layer is None and hparams['embedding_word_shape'] is None:
            raise RuntimeError("""Either custom_word_embedding_layer or embedding_word_shape must be provided!""")

        if custom_word_embedding_layer is None:
            self.word_embedding = torch.nn.Embedding(hparams['embedding_word_shape'][0],hparams['embedding_word_shape'][1], padding_idx=hparams['embedding_word_padding_idx'])
        else:
            self.word_embedding = custom_word_embedding_layer
        
        embedding_word_dim = self.word_embedding.embedding_dim
        self.word_embedding.weight.requires_grad = word_embedding_requires_grad
        
        # 2. sequence encoder phase (LSTM)

        self.lstm_word = nn.LSTM(
            input_size = embedding_word_dim, 
            hidden_size = hparams['lstm_parameters']['hidden_size'], 

            bidirectional = hparams['lstm_parameters']['bidirectional'], 
            num_layers = hparams['lstm_parameters']['num_layers'], 
            dropout = hparams['lstm_parameters']['dropout'] if hparams['lstm_parameters']['num_layers'] > 1 else 0,
            batch_first = batch_first 
        )

        lstm_word_word_output_dim = 2*hparams['lstm_parameters']['hidden_size'] if hparams['lstm_parameters']['bidirectional'] is True else hparams['lstm_parameters']['hidden_size']

        # 3. classifier (and optionally crf)

        self.fc1 = nn.Linear(lstm_word_word_output_dim, lstm_word_word_output_dim//2)
        self.fc2 = nn.Linear(lstm_word_word_output_dim//2, lstm_word_word_output_dim//4)
        self.fc3 = nn.Linear(lstm_word_word_output_dim//4  + self.n_pos, lstm_word_word_output_dim//4)
        self.classifier = nn.Linear(lstm_word_word_output_dim//4, self.n_labels)

        # if use_crf is False and loss_fn is None:
        #     raise RuntimeError("""Either use_crf = True or loss_fn must be provided!""")
        if use_crf:
            self.crf = CRF(self.n_labels) # , batch_first=batch_first)
        else:
            self.crf = None
            self.loss_fn = loss_fn

        # Extra layers repeated during the net

        self.dropout = nn.Dropout(hparams['general_dropout'])
        self.relu = nn.ReLU()

        self.to(device) # finally, put in the right device


    def compute_outputs(self, x_word, x_pos):
        ''' compute the model until the classifier layer (before CRF) '''
        
        x_word = self.word_embedding(x_word)

        x_word = self.dropout(x_word)

        x_word, _ = self.lstm_word(x_word)
        
        # x_word = self.dropout(x_word)
        
        x_word = self.fc1(x_word)
        x_word = self.dropout(x_word)
        x_word = self.relu(x_word)

        x_word = self.fc2(x_word)
        x_word = self.relu(x_word)

        x_pos_one_hot = torch.nn.functional.one_hot(x_pos, self.n_pos)
        x_word = torch.cat((x_word, x_pos_one_hot),dim=-1)

        x_word = self.fc3(x_word)
        x_word = self.dropout(x_word)
        x_word = self.relu(x_word)

        x_word = self.classifier(x_word)

        return x_word

    def compute_loss(self, x, y_true, mask = None):
        ''' compute negative log-likelihood (loss_fn() if use_crf = False). Mask is only used when use_crf = True '''
        if self.crf is None:
            return self.loss_fn(x, y_true)

        loss: torch.Tensor = -(
            torch.sum(self.crf.forward(x,y_true,mask)) / len(x)
        )

        return loss
        
    def forward(self, x_word, x_pos, mask = None):
        ''' predict labels '''
        x = self.compute_outputs(x_word, x_pos)
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
