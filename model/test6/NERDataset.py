from torch import as_tensor
from torch.utils.data import Dataset
import numpy as np

import re

from typing import List, Dict

class NERDataset(Dataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(   self,  flile_path:str, extra_flile_path:str = None, 
                    vocabulary = None, vocabulary_label = None, vocabulary_char = None,
                    params = None,
                    encode_numbers = None):
        """
        Args:
            - file_path: path to the file to open
            - extra_file_path: if passed, this dataset will be also added to data (train_extra.txt file)
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - vocabulary_label: the vocabulary dictionary for labels obtained by get_vocabulary_label(), composed of 'key_to_index' and 'index_to_key'
            - vocabulary_char: the vocabulary dictionary for characters, composed of 'key_to_index' and 'index_to_key'
            - params: the dictionary needed to initialize this class, composed of:
                - window_size: maximum size of a sentence
                - window_shift: how many steps the window will do to go to the next one
                - PAD_TOKEN: the string value for the token
                - PAD_INDEX: the index of the token for the labels, in order to be ignored by the loss
                - max_word_length: maximum length of a word
        """
        data = NERDataset.read_dataset(flile_path, encode_numbers = encode_numbers)

        self.vocabulary = vocabulary
        self.vocabulary_label = vocabulary_label if vocabulary_label is not None else NERDataset.get_vocabulary_label(data)
        self.vocabulary_char = vocabulary_char
        
        if extra_flile_path is not None:
            data = data + self.read_extra_dataset(
                            extra_flile_path, 
                            self.vocabulary_label, 
                            params['window_size'], lowered = True)

        print( 'max sentence length:' , max([len(sentence_list['inputs']) for sentence_list in data]) , 'dataset length:' , len(data))
        
        self.generate_chars_from_data(data, max_word_length = params['max_word_length'], pad_token = params['PAD_TOKEN'])

        self.data = NERDataset.generate_windows(data, 
                                                params['window_size'], 
                                                params['window_shift'], 
                                                pad_token = params['PAD_TOKEN'], 
                                                pad_index = params['PAD_INDEX'],
                                                max_word_length = params['max_word_length'])

    @staticmethod
    def read_dataset(file_path:str, encode_numbers = None):
        """
        Reads the dataset for this homework \n
        Args:
            - file_path: path to the file to open 
            - encode_numbers: if not None, the numbers are encoded in the words via the character provided (e.g. '#') \n
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
                    sentence['inputs'].append(row_parts[0] if encode_numbers is None else re.sub('\d', encode_numbers, row_parts[0]))
                    sentence['outputs'].append(row_parts[1])

        data_file.close()
        return data

    @staticmethod
    def read_extra_dataset(file_path:str, vocabulary_label, max_length = -1, lowered = False):
        """
        Reads the other extra dataset format and returns only the formatted data that match with the original dataset \n
        Args:
            - file_path: path to the file to open 
            - vocabulary_label: vocabulary to check if labels are all in vocabulary. Discard sentence if not
            - max_length: maximum length of a sentence to be added 
            - lowered: if we need to lower the word \n
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

                if len(row_parts) > 1:
                    word = row_parts[1] if lowered is False else row_parts[1].lower()
                    sentence['inputs'].append(word)
                    sentence['outputs'].append(row_parts[2])

                else:
                    if len(sentence['inputs']) > 0:
                        if max_length == -1 or len(sentence['inputs']) <= max_length:
                            if all(label in vocabulary_label['index_to_key'] for label in sentence['outputs']):
                                data.append( sentence.copy() )
                    sentence = { 'inputs':[] , 'outputs':[] }

        data_file.close()
        return data

    def generate_chars_from_data(self, data, max_word_length = -1, pad_token = None):
        """ 
        Populates data with character-level informations, transforming it in a list of dictionaries \n
        Args:
            - data: the data returned by read_dataset(), could be a list of lists or a list of dictionaries with 'inputs' key in it
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied
            - pad_token: the pad string to apply \n
        """
        for i, sentence in enumerate(data):
            data[i]['chars'] = NERDataset.generate_chars_from_sentence(sentence['inputs'], max_word_length, pad_token)
            

    @staticmethod
    def generate_chars_from_sentence(sentence, max_word_length = -1, pad_token = None):
        """
        Args:
            - sentence: a list of words
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied
            - pad_token: the pad string to apply \n
        Returns:
            list of lists of chararcters
        """
        return [(list(word)[0:max_word_length] if max_word_length > 0 else list(word)) + [pad_token]*(max_word_length - len(word)) for word in sentence]


    @staticmethod
    def get_vocabulary_label(data):
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
        return { 'key_to_index':label2id , 'index_to_key':id2label }

    @staticmethod
    def generate_windows(data, window_size, window_shift, pad_token = None, pad_index = -1, max_word_length = -1):
        """
        Args:
            - data: the data returned by read_dataset()
            - window_size: the maximum size of a sentence. Bigger sentences generates two or more windows
            - window_shift: the amount of shift from the last window of the same sentence to make. if it's equal to window_size then the current window starts after the end of the other
            - pad_token: the string format of the token (could be also None)
            - pad_index: the padding index for labels (used to remove them from the loss function) 
            - max_word_length: max length of a word before padding is applied. If < 0, then padding is never applied \n
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
    def prepare_batch_fn(vocabulary, vocabulary_label, vocabulary_char, unk_tok):
        """
        Args:
            - vocabulary: the vocabulary dictionary obtained by gensim, composed of 'key_to_index' and 'index_to_key' for the words
            - vocabulary_label: the vocabulary dictionary obtained by get_vocabulary_label(), composed of 'key_to_index' and 'index_to_key'
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
            outputs = [NERDataset.encode_sentence_labels(sample, vocabulary_label) for sample in outputs]
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
            the encoded version of the sentence, represented by numbers
        """
        encoded = [vocabulary['key_to_index'][word] if word in vocabulary['key_to_index'] else vocabulary['key_to_index'][unk_tok] for word in sentence]
        return encoded

    @staticmethod
    def encode_sentence_labels(sentence_labels, vocabulary_label):
        """
        Args:
            - sentence: a list of strings (labels)
            - vocabulary_label: the vocabulary dictionary obtained by get_vocabulary_label(), composed of 'key_to_index' and 'index_to_key' \n
        Returns:
            the encoded version of the labels, represented by numbers
        """
        encoded = [vocabulary_label['key_to_index'][label] if label in vocabulary_label['key_to_index'] else label for label in sentence_labels]
        return encoded

    @staticmethod
    def encode_sentence_chars(sentence_chars, vocabulary_char, unk_tok):
        """
        Args:
            - sentence_chars: a list of lists of charaters, e.g. [ ['T','h','e','<pad>'],['b','e','s','t'] ]
            - vocabulary_char: the vocabulary dictionary, composed of 'key_to_index' and 'index_to_key' for the chars
            - unk_token: the string representation of the unknown token \n
        Returns:
            the encoded version of the chars, represented by numbers
        """
        encoded_chars = []
        for word_list in sentence_chars:
            encoded_chars_word = [vocabulary_char['key_to_index'][char] if char in vocabulary_char['key_to_index'] else vocabulary_char['key_to_index'][unk_tok] for char in word_list]
            encoded_chars.append( encoded_chars_word )
        return encoded_chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save_vocabulary_char(self, path):
        np.save(path, self.vocabulary_char)

    def save_vocabulary_label(self, path):
        np.save(path, self.vocabulary_label)

    def save_vocabulary(self, path):
        np.save(path, self.vocabulary)
    
    @staticmethod
    def load_vocabulary_char(path):
        return np.load(path, allow_pickle=True).tolist()

    @staticmethod
    def load_vocabulary_label(path):
        return np.load(path, allow_pickle=True).tolist()

    @staticmethod
    def load_vocabulary(path):
        return np.load(path, allow_pickle=True).tolist()