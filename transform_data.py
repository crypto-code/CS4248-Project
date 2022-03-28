import json
from tqdm import tqdm
import re
import pandas as pd

label2id = {
    "O": 0,
    "Task": 1,
    "Process": 2,
    "Material": 3
}


def load_data(fname):
    with open(fname) as f:
        json_data = json.load(f)

    X = []
    Y = []

    print('Loading Dataset...')

    for file, data in tqdm(json_data.items()):
        X.append(data["text"])
        Y.append(data["keywords"])

    return X, Y


def get_word_index(word_list, start, end):
    start_ind = 0
    end_ind = 0
    pos = -1
    for ind, word in enumerate(word_list):
        for c in word:
            if pos == start:
                start_ind = ind
            if pos == end:
                end_ind = ind
            pos += 1

    return start_ind, end_ind


def clean_text(text):
    text = re.sub(r'\[.+\]', '', text)
    return text


def generate_all_possible(sequences):
    all_possible = []
    for i in range(len(sequences)):
        for j in range(len(sequences) - i):
            all_possible.append(' '.join(x for x in sequences[j:j + i]).strip())
    return all_possible


X, Y = load_data("./json_data.json")

dataset = []

for x, y in zip(X, Y):
    sentences = generate_all_possible(re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", clean_text(x)))
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        tokenized_x = re.split(r"\s+", sentence)
        datapoint = {"text": sentence}
        output_y = [0] * (len(tokenized_x))
        for keyword, details in y.items():
            groups = re.finditer(r"\b" + keyword + r"\b", sentence)
            if groups is None:
                continue
            for count, match in enumerate(groups):
                count_idx = min(len(details) - 1, count)
                start, end = match.start(), match.end()
                start -= sentence.count(" ", 0, start)
                end -= sentence.count(" ", 0, end + 1)
                start_index, end_index = get_word_index(tokenized_x, start, end)
                for i in range(start_index, end_index):
                    output_y[i] = label2id[details[min(count_idx, 0)][2]]
        datapoint["labels"] = output_y
        dataset.append(datapoint)

dataset = pd.DataFrame(dataset)
dataset.to_csv("./wikitdata.csv")
