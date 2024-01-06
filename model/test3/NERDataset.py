
from torch import as_tensor
from torch.utils.data import Dataset
import numpy as np

from typing import List, Dict

class NERDataset(Dataset):
    def __init__(self, flile_path:str, vocabulary = None, params = None):
        """
        Args:
            - file_path: path to the file to open
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - params: the dictionary needed to initialize this class (see notebook nlp-hw1_test2 for more details)
        """
        self.vocabulary = vocabulary
        data = NERDataset.read_dataset(flile_path)
        [self.label2id, self.id2label] = self.create_label_mapping(data) if vocabulary is not None else [None, None]
        self.data = NERDataset.generate_windows(data, params['window_size'], params['window_shift'], pad_token = params['PAD_TOKEN'], pad_index = params['PAD_INDEX'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def read_dataset(file_path:str):
        """
        Args:
            - file_path: path to the file to open \n
        Returns:
            list of dictionaries, each dictionary has: \n
                'inputs': sentence in list format, e.g. ['this','is',...] \n
                'outputs': labels of the sentence in list format, e.g. ['O','O',...]
        """
        data = []
        sentence = { 'inputs':[] , 'outputs':[] }
        with open(file_path, "r") as data_file:
            for row in data_file:

                row_parts = row.rstrip().split('\t')

                if row_parts[0] == '#' and row_parts[1] == 'id':
                    if len(sentence['inputs']) > 0:
                        data.append( sentence.copy() )
                    sentence = { 'inputs':[] , 'outputs':[] }
                    continue

                if len(row_parts) > 1:
                    sentence['inputs'].append(row_parts[0])
                    sentence['outputs'].append(row_parts[1])

        data_file.close()
        return data

    @staticmethod
    def generate_windows(data, window_size, window_shift, pad_token = None, pad_index = -1):
        """
        Args:
            - data: the data returned by read_dataset()
            - window_size: the maximum size of a sentence. Bigger sentences generates two or more windows
            - window_shift: the amount of shift from the last window of the same sentence to make. if it's equal to window_size then the current window starts after the end of the other
            - pad_token: the string format of the token (could be also None)
            - pad_index: the padding index for labels (used to remove them from the loss function) \n
        Returns:
            list of dictionaries, each dictionary has: \n
                'inputs': window in list format, e.g. ['this','is',...] \n
                'outputs': labels of the sentence in list format, e.g. ['O','O',...]
        """
        windowed_data = []
        for sentence in data:
            windowed_inputs = [sentence['inputs'][i:i+window_size] for i in range(0, len(sentence['inputs']), window_shift)]
            windowed_outputs = [sentence['outputs'][i:i+window_size] for i in range(0, len(sentence['outputs']), window_shift)]

            for window_input, window_output in zip(windowed_inputs, windowed_outputs):
                window_input = window_input + [pad_token]*(window_size - len(window_input))
                window_output = window_output + [pad_index]*(window_size - len(window_output))
                windowed_data.append({ 'inputs':window_input , 'outputs':window_output })

        return windowed_data

    @staticmethod
    def generate_windows_sentence(data, window_size, window_shift, pad_token = None, pad_index = -1):
        """
        Args:
            - data: it is a list of lists of strings (words)
            - window_size: the maximum size of a sentence. Bigger sentences generates two or more windows
            - window_shift: the amount of shift from the last window of the same sentence to make. if it's equal to window_size then the current window starts after the end of the other
            - pad_token: the string format of the token (could be also None).
            - pad_index: the padding index for labels (used to remove them from the loss function) \n
        Returns:
            list of windows of strings (words)
        """
        windowed_data = []
        for sentence in data:
            windowed_inputs = [sentence[i:i+window_size] for i in range(0, len(sentence), window_shift)]

            for window_input in windowed_inputs:
                window_input = window_input + [pad_token]*(window_size - len(window_input))
                windowed_data.append(window_input)

        return windowed_data

    @staticmethod
    def prepare_batch_fn(vocabulary, label2id, unk_tok):
        """
        Args:
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - label2id: the dictionary that converts a label to its id
            - unk_token: the string representation of the unknown token \n
        Returns:
            the function necessary for the collate_fn parameter in the torch.utils.data.DataLoader
        """
        def prepare_batch(batch: List[Dict]) -> List[Dict]:
            # extract sentences and labels not encoded
            inputs = [sample['inputs'] for sample in batch]
            outputs = [sample['outputs'] for sample in batch]
            # encode them
            inputs = [NERDataset.encode_sentence_words(sample, vocabulary, unk_tok) for sample in inputs]
            outputs = [NERDataset.encode_sentence_labels(sample, label2id) for sample in outputs]
        
            return { 'inputs': as_tensor(inputs) , 'outputs': as_tensor(outputs) }
        return prepare_batch

    @staticmethod
    def encode_sentence_words(sentence, vocabulary, unk_tok):
        """
        Args:
            - sentence: a list of strings (words)
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - unk_token: the string representation of the unknown token \n
        Returns:
            the encoded version of the sentence
        """
        encoded = [vocabulary['key_to_index'][word] if word in vocabulary['key_to_index'] else vocabulary['key_to_index'][unk_tok] for word in sentence]
        return encoded

    @staticmethod
    def encode_sentence_labels(sentence_labels, label2id):
        """
        Args:
            - sentence: a list of strings (labels)
            - label2id: the dictionary that converts a label to its id \n
        Returns:
            the encoded version of the sentence
        """
        encoded = [label2id[label] if label in label2id else label for label in sentence_labels]
        return encoded

    @staticmethod
    def create_label_mapping(data):
        """
        Args:
            - data: the data returned by read_dataset()
        Returns:
            the mapping from label to id and from id to label, respectively
        """
        labels = set()
        for sentence in data:
            for value in sentence['outputs']:
                labels.add(value)
        labels = list(labels)
        labels.sort()
        label2id = {l:i for i,l in enumerate(labels)}
        id2label = labels
        return [label2id, id2label]

    def save_labels(self, path):
        np.save(path, [self.label2id, self.id2label])

    def save_vocabulary(self, path):
        np.save(path, self.vocabulary)

    @staticmethod
    def load_labels(path):
        [label2id, id2label] = np.load(path, allow_pickle=True).tolist()
        return [label2id, id2label]

    @staticmethod
    def load_vocabulary(path):
        return np.load(path, allow_pickle=True).tolist()
