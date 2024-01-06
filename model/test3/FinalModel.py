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
        model_save_file_name = 'nernet_weights.pth',
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('Creating model...')
        self.model = NERNet(    embedding_shape=self.globalParams['embedding_shape'], 
                                custom_embedding_layer = custom_embedding_layer,
                                seq_encoder_parameters=self.globalParams['seq_encoder_parameters'], 
                                n_labels=self.globalParams['n_labels'], 
                                use_crf=self.globalParams['use_crf'],
                                loss_fn=None if self.globalParams['use_crf'] else loss_fn,
                                device=self.device)
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

        # if the window_size is 10 and the window_shift is 2, then
        # the discriminant is (window_size - window_shift)//2 = 4. So
        # from the first window will be used only from 0 to 6th label, 
        # from the second window will be used only from 7th to 12th label. 
        # The value of window_size - window_shift MUST be divisible by 2!
        discriminant = (self.globalParams['window_size'] - self.globalParams['window_shift']) // 2

        self.model.eval()
        with torch.no_grad():
            for sentence in tokens:
                windows = NERDataset.generate_windows_sentence(
                                        [sentence], # [[w.lower() for w in sentence]], 
                                        self.globalParams['window_size'], self.globalParams['window_shift'],
                                        pad_token = self.globalParams['PAD_TOKEN'],
                                        pad_index = self.globalParams['PAD_INDEX'])

                for i in range(len(windows)):
                    windows[i] = NERDataset.encode_sentence_words(windows[i], self.vocabulary, self.globalParams['UNK_TOKEN'])
                
                windows = torch.as_tensor(windows).to(self.device)
                mask = ~windows.eq( self.vocabulary['key_to_index'][self.globalParams['PAD_TOKEN']] ).to(self.device)
                prediction = self.model(windows, mask)

                if self.model.crf is None:
                    res_pred = self.model.get_indices( prediction ).cpu().numpy().tolist()
                else:
                    res_pred = prediction

                # applying 'discriminant' logic...
                for i in range(len(res_pred)):
                    res_pred[i] = res_pred[i][ (i>0)*discriminant : self.globalParams['window_size'] - (i+1!=len(res_pred))*discriminant ]
                
                res_pred = [item for window in res_pred for item in window][:len(sentence)] # concatenating and removing padding

                res_pred = [self.id2label[w] for w in res_pred]
                predictions.append(res_pred)
        return predictions