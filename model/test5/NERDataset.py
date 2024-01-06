from torch import as_tensor
from torch.utils.data import Dataset
import numpy as np

from typing import List, Dict

class NERDataset(Dataset):
    def __init__(self,  flile_path:str, vocabulary = None, vocabulary_char = None, params = None):
        """
        Args:
            - file_path: path to the file to open
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - vocabulary_char: the vocabulary dictionary for character-level, composed of 'key_to_index' and 'index_to_key' for the words
            - params: the dictionary needed to initialize this class (see notebook nlp-hw1_test2 for more details)
        """
        self.vocabulary = vocabulary
        self.vocabulary_char = vocabulary_char

        data = NERDataset.read_dataset(flile_path)
        self.generate_chars_from_data(data, params['max_word_length'], params['PAD_TOKEN'])
        
        [self.label2id, self.id2label] = self.create_label_mapping(data) if vocabulary is not None else [None, None]

        self.data = NERDataset.generate_windows(data, 
                                                params['window_size'], 
                                                params['window_shift'], 
                                                max_word_length = params['max_word_length'],
                                                pad_token = params['PAD_TOKEN'], 
                                                pad_index = params['PAD_INDEX'])
        
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
    def generate_chars_from_data(data, max_word_length = -1, pad_token = None):
        """ 
        Populates data with character-level informations, transforming it in a list of dictionaries
        Args:
            - data: the data returned by read_dataset(), could be a list of lists or a list of dictionaries with 'inputs' key in it
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied
            - pad_token: the pad string to apply \n
        """
        for i, sentence in enumerate(data):
            sentence = {'inputs':sentence} if type(sentence) is not dict else sentence
            sentence['chars'] = NERDataset.generate_chars_from_sentence(sentence['inputs'], max_word_length, pad_token)
            data[i] = sentence

    @staticmethod
    def generate_chars_from_sentence(sentence, max_word_length = -1, pad_token = None):
        """
        Args:
            - sentence: a list of words
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied
            - pad_token: the pad string to apply
        Returns:
            list of lists of chararcters
        """
        return [(list(word)[0:max_word_length] if max_word_length > 0 else list(word)) + [pad_token]*(max_word_length - len(word)) for word in sentence]

    @staticmethod
    def generate_windows(data, window_size, window_shift, max_word_length = -1, pad_token = None, pad_index = -1):
        """
        Args:
            - data: the data returned by read_dataset()
            - window_size: the maximum size of a sentence. Bigger sentences generates two or more windows
            - window_shift: the amount of shift from the last window of the same sentence to make. if it's equal to window_size then the current window starts after the end of the other
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied
            - pad_token: the string format of the token (could be also None)
            - pad_index: the padding index for labels (used to remove them from the loss function) \n
        Returns:
            list of dictionaries, each dictionary has: \n
                'inputs': window in list format, e.g. ['this','is',...] 
                'outputs': labels of the sentence in list format, e.g. ['O','O',...] 
                'chars': list of lists of characters, e.g. [['t','h','i','s'],...] 
        """
        windowed_data = []
        for sentence in data:
            windowed_inputs = [sentence['inputs'][i:i+window_size] for i in range(0, len(sentence['inputs']), window_shift)]
            windowed_outputs = [sentence['outputs'][i:i+window_size] for i in range(0, len(sentence['outputs']), window_shift)]
            windowed_chars = [sentence['chars'][i:i+window_size] for i in range(0, len(sentence['chars']), window_shift)]
            
            for window_input, window_output, window_chars in zip(windowed_inputs, windowed_outputs, windowed_chars):
                window_input = window_input + [pad_token]*(window_size - len(window_input))
                window_output = window_output + [pad_index]*(window_size - len(window_output))
                window_chars = window_chars + [[pad_token]*max_word_length]*(window_size - len(window_chars))

                windowed_data.append({ 'inputs':window_input , 'outputs':window_output , 'chars':window_chars })

        return windowed_data

    @staticmethod
    def generate_windows_sentence(  data, window_size, window_shift, 
                                    pad_token = None, pad_index = -1, vocabulary_pos = None, 
                                    pos_tagger = None,
                                    max_word_length = -1):
        """
        Args:
            - data: it is a list of lists of strings (words)
            - window_size: the maximum size of a sentence. Bigger sentences generates two or more windows
            - window_shift: the amount of shift from the last window of the same sentence to make. if it's equal to window_size then the current window starts after the end of the other
            - pad_token: the string format of the token (could be also None).
            - pad_index: the padding index for labels (used to remove them from the loss function) 
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied \n
        Returns:
            list of dictionaries, each dictionary has: \n
                'inputs': window in list format, e.g. ['this','is',...]
                'chars': list of lists of characters, e.g. [['t','h','i','s'],...] 
        """
        windowed_data = []
        NERDataset.generate_chars_from_data(data, max_word_length, pad_token)
        for sentence in data:
            windowed_inputs = [sentence['inputs'][i:i+window_size] for i in range(0, len(sentence['inputs']), window_shift)]
            windowed_chars = [sentence['chars'][i:i+window_size] for i in range(0, len(sentence['chars']), window_shift)]

            for window_input, window_chars in zip(windowed_inputs, windowed_chars):
                window_input = window_input + [pad_token]*(window_size - len(window_input))
                window_chars = window_chars + [[pad_token]*max_word_length]*(window_size - len(window_chars))

                windowed_data.append({ 'inputs':window_input , 'chars':window_chars })

        return windowed_data

    @staticmethod
    def prepare_batch_fn(vocabulary, label2id, vocabulary_char, unk_tok):
        """
        Args:
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - label2id: the dictionary that converts a label to its id
            - vocabulary_char: the vocabulary dictionary for character-level, composed of 'key_to_index' and 'index_to_key' for the words
            - unk_token: the string representation of the unknown token \n
        Returns:
            the function necessary for the collate_fn parameter in the torch.utils.data.DataLoader
        """
        def prepare_batch(batch: List[Dict]) -> List[Dict]:
            # extract sentences and labels not encoded
            inputs = [sample['inputs'] for sample in batch]
            outputs = [sample['outputs'] for sample in batch]
            chars = [sample['chars'] for sample in batch]
            # encode them
            inputs = [NERDataset.encode_sentence_words(sample, vocabulary, unk_tok) for sample in inputs]
            outputs = [NERDataset.encode_sentence_labels(sample, label2id) for sample in outputs]
            chars = [NERDataset.encode_sentence_chars(sample, vocabulary_char, unk_tok) for sample in chars]
        
            return { 'inputs': as_tensor(inputs) , 'outputs': as_tensor(outputs) , 'chars': as_tensor(chars) }
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
    def encode_sentence_chars(sentence_chars, vocabulary_char, unk_tok):
        """
        Args:
            - sentence_chars: a list of lists of charaters, e.g. [ ['T','h','e','<pad>'],['b','e','s','t'] ]
            - vocabulary_char: the vocabulary dictionary, composed of 'key_to_index' and 'index_to_key' for the chars
            - unk_token: the string representation of the unknown token \n
        Returns:
            the encoded version of the chars
        """
        encoded_chars = []
        for word_list in sentence_chars:
            encoded_chars_word = [vocabulary_char['key_to_index'][char] if char in vocabulary_char['key_to_index'] else vocabulary_char['key_to_index'][unk_tok] for char in word_list]
            encoded_chars.append( encoded_chars_word )
        return encoded_chars

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

    def save_vocabulary_char(self, path):
        np.save(path, self.vocabulary_char)

    @staticmethod
    def load_labels(path):
        [label2id, id2label] = np.load(path, allow_pickle=True).tolist()
        return [label2id, id2label]

    @staticmethod
    def load_vocabulary(path):
        return np.load(path, allow_pickle=True).tolist()

    @staticmethod
    def load_vocabulary_char(path):
        return np.load(path, allow_pickle=True).tolist()
