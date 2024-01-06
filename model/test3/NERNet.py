# this model parameters and description could change in test4

import torch.nn as nn
import torch
import numpy as np

from TorchCRF import CRF

class NERNet(nn.Module):
    def __init__(self,  custom_embedding_layer = None, embedding_shape = None, freeze_embedding = True, 
                        seq_encoder_parameters = [
                            {'hidden_size':80,'bidirectional':True, 'num_layers':2, 'dropout':0.2},
                        ],
                        n_labels = 13,
                        use_crf = False,
                        loss_fn = None,
                        batch_first = True, # ! remember
                        device = 'cpu'):
        """
        Args:
            - custom_embedding_layer: if not None, it will be used as embedding layer, otherwise a new embedding will be generated with shape embedding_shape = (vocab_dim, embedding_dim)
            - freeze_embedding: if True, the embedding weights will not change during training
            - seq_encoder_parameters: a list of dictionary to create as many LSTMs as the length of the list, in which each dictionary has:
                - hidden_size: the hidden size of the (i-th) LSTM
                - bidirectional: if the (i-th) LSTM is bidirectional or not
                - num_layers: number of layers of the (i-th) LSTM
                - dropout: dropout value, between 0 and 1
            - n_labels: number of labels
            - use_crf: if the CRF layer must be generated. If not, loss_fn must be provided
            - loss_fn: the loss to be used. Ignored if use_crf is True. It must be used in this format: loss_function(predictions, labels)
            - batch_first: ff True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            - device: the device that the model needs to be attached to
        """
        super().__init__()

        self.n_labels = n_labels
        self.device = device
        embedding_requires_grad = not freeze_embedding
        
        # 1. create the encoder if custom embedding is not 

        if custom_embedding_layer is None and embedding_shape is None:
            raise RuntimeError("""Either custom_embedding_layer or embedding_shape must be provided!""")

        if custom_embedding_layer is None:
            self.embedding = torch.nn.Embedding(embedding_shape[0],embedding_shape[1])
            embedding_dim = embedding_shape[1]
        else:
            self.embedding = custom_embedding_layer
            embedding_dim = custom_embedding_layer.embedding_dim
            
        self.embedding.weight.requires_grad = embedding_requires_grad
        
        # 2. sequence encoder(s) phase (LSTMs)

        self.seq_encoder = nn.ModuleList()
        if len(seq_encoder_parameters) < 1:
            raise RuntimeError("""There must be at least one layer of sequence encoders (LSTMs)!""")
        last_input_size = embedding_dim
        for i, p in enumerate(seq_encoder_parameters):
            layer_lstm = nn.LSTM(
                input_size = last_input_size, 
                hidden_size = p['hidden_size'], 
                bidirectional = p['bidirectional'], 
                num_layers = p['num_layers'], 
                dropout = p['dropout'],
                batch_first = batch_first 
            )
            last_input_size = 2*p['hidden_size'] if p['bidirectional'] else p['hidden_size']
            self.seq_encoder.append(layer_lstm)


        # 3. classifier (and optionally crf)

        if use_crf is False and loss_fn is None:
            raise RuntimeError("""Either use_crf = True or loss_fn must be provided!""")

        self.classifier = nn.Linear(last_input_size, n_labels)

        if use_crf:
            self.crf = CRF(n_labels) # , batch_first=batch_first)
        else:
            self.crf = None
            self.loss_fn = loss_fn

        self.to(device)

    def compute_outputs(self, x):
        ''' compute the model until the classifier layer '''
        x = self.embedding(x)
        for i, lstm_layer in enumerate(self.seq_encoder):
            x, h = self.seq_encoder[i](x)
        x = self.classifier(x)
        return x

    def compute_loss(self, x, y_true, mask = None):
        ''' compute negative log-likelihood (loss_fn() if use_crf = False). Mask is only used when use_crf = True '''
        if self.crf is None:
            return self.loss_fn(x, y_true)

        loss: torch.Tensor = -(
            torch.sum(
                self.crf.forward(
                    x,
                    y_true,
                    mask,
                )
            )
            / len(x)
        )

        return loss
        
    def forward(self, x, mask = None):
        ''' predict labels '''
        x = self.compute_outputs(x)
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
