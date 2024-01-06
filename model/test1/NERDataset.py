
from torch.utils.data import Dataset
import numpy as np

PAD_TOKEN = '<pad>'
window_size = 44
window_shift = window_size

class NERDataset(Dataset):
    def __init__(self, flile_path:str, window_size:int, pad_token:str, window_shift:int = -1):
        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        self.pad_token = pad_token
        self.ignore_index = -1
        
        data = NERDataset.read_dataset(flile_path)
        self.create_label_mapping(data)
        windowed_data = NERDataset.generate_windows(data, self.window_size, self.window_shift, self.generate_pad_element())
        self.encoded_data = self.encode_data(windowed_data)
        
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Data is not encoded yet!""")
        return self.encoded_data[idx]
    
    @staticmethod
    def read_dataset(file_path:str):
        data = []
        sentence = []
        with open(file_path, "r") as data_file:
            for row in data_file:

                row_parts = row.rstrip().split('\t')

                if row_parts[0] == '#' and row_parts[1] == 'id':
                    if len(sentence) > 0:
                        data.append( sentence )
                    sentence = []
                    continue

                if len(row_parts) > 1:
                    sentence.append( { 'word': row_parts[0] , 'tag':  row_parts[1]} )

        data_file.close()
        return data
    
    def create_label_mapping(self, data):
        labels = set()
        for sentence in data:
            for value in sentence:
                labels.add(value['tag'])
        labels = list(labels)
        labels.sort()
        self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = labels

    def save_labels(self, path):
        np.save(path, [self.label2id, self.id2label])

    @staticmethod
    def load_labels(path):
        [label2id, id2label] = np.load(path, allow_pickle=True)
        return [label2id, id2label]
    
    @staticmethod
    def generate_windows(data, window_size, window_shift, pad_element = None):
        windowed_data = []
        for sentence in data:
            windowed_sentences = [sentence[i:i+window_size] for i in range(0, len(sentence), window_shift)]
            for window in windowed_sentences:
                window = window + [pad_element]*(window_size - len(window))
                windowed_data.append(window)
            if pad_element not in windowed_data[-1]: # Adding additional None values as dummy sentence in order to determine the end of a sentence
                windowed_data.append([pad_element]*window_size)
        return windowed_data
    
    def generate_pad_element(self):
        return { 'word': self.pad_token , 'tag':  self.ignore_index} # tag is -1, to be ignored by the loss and evaluation method!!!!
    
    def encode_data(self, data):
        encoded_data = []
        for window in data:
            encoded_inputs = []
            encoded_outputs = []
            for word in window:
                encoded_inputs.append( word['word'] )
                tag = word['tag'] if word['word'] == self.pad_token else self.label2id[word['tag']]
                encoded_outputs.append( tag )
            encoded_data.append( { 'inputs':encoded_inputs , 'outputs':encoded_outputs } )
        return encoded_data
