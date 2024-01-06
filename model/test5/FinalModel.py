import numpy as np
import torch

from os.path import join

from NERDataset import NERDataset
from NERNet import NERNet

from typing import List, Tuple

class FinalModel():
    def __init__(   
        self, 
        saves_path, 
        model_save_file_name = 'nernet_weights_simple.pth',
        load_model = True, 
        custom_embedding_layer = None, 
        loss_fn = None
    ):
        """
        Args:
            - saves_path: path to all weights and variables needed for the model
            - model_save_file_name: the name (with extension) of the saved network model weights
            - load_model: if true, loads the model network
            - embedding_layer: if not none, the custom embedding layer will be passed to the model
            - loss_fn: the loss function to use for the model (ignored if use_crf = True)
        """
        print('Creating final model...')
        self.globalParams = np.load(join(saves_path,'global_params.npy'), allow_pickle=True).tolist()
        [self.label2id, self.id2label] = NERDataset.load_labels(join(saves_path,'dataset_labels.npy'))
        self.vocabulary = NERDataset.load_vocabulary(join(saves_path,'dataset_vocabulary.npy'))

        self.vocabulary_char = NERDataset.load_vocabulary(join(saves_path,'dataset_vocabulary_char.npy'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('Creating model...')
        self.model = NERNet(hparams = self.globalParams,
                            custom_word_embedding_layer = custom_embedding_layer,
                            custom_char_embedding_layer = None,
                            loss_fn = None if self.globalParams['use_crf'] else loss_fn,
                            device = self.device)

        if load_model:
            print('Loading model weights...')
            try:
                self.model.load_weights(join(saves_path,model_save_file_name))
            except:
                print('Loading model weights failed with strict=True, trying with False...')
                self.model.load_weights(join(saves_path,model_save_file_name), strict=False)
        self.model.to(self.device)
        print('Init done')

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for sentence in tokens:

                sentence_in = NERDataset.encode_sentence_words(sentence, self.vocabulary, self.globalParams['UNK_TOKEN'])

                sentence_chars = NERDataset.generate_chars_from_sentence(sentence, self.globalParams['max_word_length'], self.globalParams['PAD_TOKEN'])
                sentence_chars = NERDataset.encode_sentence_chars(sentence_chars, self.vocabulary_char, self.globalParams['UNK_TOKEN'])

                sentence_in = torch.as_tensor([sentence_in]).to(self.device)
                mask = ~sentence_in.eq( self.vocabulary['key_to_index'][self.globalParams['PAD_TOKEN']] ).to(self.device)

                sentence_chars = torch.as_tensor([sentence_chars]).to(self.device)

                prediction = self.model(sentence_in, sentence_chars, mask)[0]

                if self.model.crf is None:
                    res_pred = self.model.get_indices( prediction ).cpu().numpy().tolist()[:len(sentence)]
                else:
                    res_pred = prediction

                res_pred = [self.id2label[w] for w in res_pred]
                
                predictions.append(res_pred)
                
        return predictions