import json
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd


def load_data(fname):
    with open(fname) as f:
        json_data = json.load(f)

    X = []
    Y = []

    print('Loading Dataset...')

    for file, data in tqdm(json_data.items()):
        X.append(data["lem_text"])
        Y.append(data["lem_keywords_word_offset"])

    return X, Y


def get_word_index(word_list, start, end):
    start_ind = 0
    end_ind = 0
    pos = -1
    for ind, word in enumerate(word_list):
        temp = word.replace("#", "")
        for c in temp:
            if pos == start:
                start_ind = ind
            if pos == end:
                end_ind = ind
            pos += 1

    return start_ind, end_ind


class CustomDataset(Dataset):
    def __init__(self, fname):
        self.data = pd.read_parquet(fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret_dict = self.data.iloc[idx].to_dict()
        for k in ret_dict.keys():
            ret_dict[k] = ret_dict[k].tolist()  #[int(x) for x in ret_dict[k].decode().strip('][').split(', ')]
        #print(ret_dict)
        return ret_dict
