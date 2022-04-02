import os.path

from transformers import BertTokenizer
from utils import load_data, get_word_index
import re
import pandas as pd

label2id = {
    "O": 0,
    "Task": 1,
    "Process": 2,
    "Material": 3
}


def generate_all_possible(sequences):
    all_possible = []
    for i in range(len(sequences)):
        for j in range(len(sequences) - i):
            all_possible.append((' '.join(x for x in sequences[j:j + i])).strip())
    return all_possible


tokenizer = BertTokenizer.from_pretrained('./bert-it')

# Process Training Data

X, Y = load_data("../json_data.json")

dataset = []

for x, y in zip(X, Y):
    sentences = generate_all_possible(re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", x))
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        tokenized_x = tokenizer.tokenize(sentence)
        prepped = []
        i = 0
        for a in tokenized_x:
            if "#" in a:
                prepped[i - 1] += a.replace("#", "")
            else:
                prepped.append(a)
                i += 1
        prepped_x = ' '.join(a for a in prepped)
        datapoint = tokenizer.encode_plus(sentence)
        output_y = [0] * (len(datapoint["input_ids"]) - 2)
        for keyword, details in y.items():
            try:
                groups = re.finditer(r"\b" + keyword + r"\b", prepped_x)
            except:
                continue
            if groups is None:
                continue
            for count, match in enumerate(groups):
                count_idx = min(len(details) - 1, count)
                start, end = match.start(), match.end()
                start -= prepped_x.count(" ", 0, start)
                end -= prepped_x.count(" ", 0, end + 1)
                start_ind, end_ind = get_word_index(tokenized_x, start, end)
                for i in range(start_ind, end_ind):
                    output_y[i] = label2id[details[min(count_idx, 0)][2]]
        datapoint = tokenizer.encode_plus(sentence)
        datapoint["labels"] = [-100] + output_y + [-100]
        dataset.append(datapoint)

dataset = pd.DataFrame(dataset)
if not os.path.exists("./data"):
    os.makedirs("./data")
dataset.to_parquet("./data/train-data.parquet")
dataset.to_csv("./data/train-wikidata.csv")

# Process Test Data

X, Y = load_data("../json_data_test.json")
dataset = []
for x, y in zip(X, Y):
    sentences = generate_all_possible(re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", x))
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        tokenized_x = tokenizer.tokenize(sentence)
        prepped = []
        i = 0
        for a in tokenized_x:
            if "#" in a:
                prepped[i - 1] += a.replace("#", "")
            else:
                prepped.append(a)
                i += 1
        prepped_x = ' '.join(a for a in prepped)
        datapoint = tokenizer.encode_plus(sentence)
        output_y = [0] * (len(datapoint["input_ids"]) - 2)
        for keyword, details in y.items():
            try:
                groups = re.finditer(r"\b" + keyword + r"\b", prepped_x)
            except:
                continue
            if groups is None:
                continue
            for count, match in enumerate(groups):
                count_idx = min(len(details) - 1, count)
                start, end = match.start(), match.end()
                start -= prepped_x.count(" ", 0, start)
                end -= prepped_x.count(" ", 0, end + 1)
                start_ind, end_ind = get_word_index(tokenized_x, start, end)
                for i in range(start_ind, end_ind):
                    output_y[i] = label2id[details[min(count_idx, 0)][2]]
        datapoint = tokenizer.encode_plus(sentence)
        datapoint["labels"] = [-100] + output_y + [-100]
        dataset.append(datapoint)

dataset = pd.DataFrame(dataset)
if not os.path.exists("./data"):
    os.makedirs("./data")
dataset.to_parquet("./data/test-data.parquet")
dataset.to_csv("./data/test-wikidata.csv")