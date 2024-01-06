import numpy as np
import torch

from os.path import join

from NERDataset import NERDataset
from NERNet import NERNet

from typing import List, Tuple

class FinalModel():
    def __init__(self, saves_path):
        print('Creating student...')
        self.globalParams = np.load(join(saves_path,'global_params.npy'), allow_pickle=True).tolist()
        [self.label2id, self.id2label] = NERDataset.load_labels(join(saves_path,'dataset_labels.npy'))
        self.vocabulary = NERDataset.load_vocabulary(join(saves_path,'dataset_vocabulary.npy'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('Loading model...')
        (vocab_dim, embedding_dim) =  self.globalParams['embedding_shape']
        self.model = NERNet(vocab_dim = vocab_dim, embedding_dim = embedding_dim, device = self.device)
        self.model.load_weights(join(saves_path,'nernet_weights.pth'))
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
                                        [[w.lower() for w in sentence]], 
                                        self.globalParams['window_size'], self.globalParams['window_shift'],
                                        pad_token = self.globalParams['PAD_TOKEN'],
                                        pad_index = self.globalParams['PAD_INDEX'])

                for i in range(len(windows)):
                    windows[i] = NERDataset.encode_sentence_words(windows[i], self.vocabulary, self.globalParams['UNK_TOKEN'])
                
                prediction = self.model(torch.as_tensor(windows).to(self.device))

                res_pred = self.model.get_indices( prediction ).cpu().numpy().tolist()
                # applying 'discriminant' logic...
                for i in range(len(res_pred)):
                    res_pred[i] = res_pred[i][ (i>0)*discriminant : self.globalParams['window_size'] - (i+1!=len(res_pred))*discriminant ]
                
                res_pred = [item for window in res_pred for item in window][:len(sentence)] # concatenating and removing padding

                res_pred = [self.id2label[w] for w in res_pred]
                predictions.append(res_pred)
        return predictions