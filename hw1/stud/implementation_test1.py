import numpy as np
from typing import List, Tuple

from model import Model

import sys
sys.path.append("./model/test1")

from NERDataset import NERDataset
from NERNet import NERNet
from gensim.models import FastText
import torch

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()
    return StudentModel()


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-CORP"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self):
        #print('Creating student...')
        [self.label2id, self.id2label] = NERDataset.load_labels('./model/test1/saves/dataset_labels.npy')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print('Creating model...')
        self.model = NERNet( FastText.load('./model/architectures/fasttext/embedding.model') , output_size=len(self.id2label) , device=device)
        #print('Loading model...')
        self.model.load_weights('./model/test1/saves/nernet_weights.pth')
        #print('Init done')

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        #print('Predicting...')
        predictions = self.model.predict(tokens)
        for i in range(len(predictions)):
           for j in range(len(predictions[i])):
               predictions[i][j] = self.id2label[predictions[i][j]]
        return predictions
