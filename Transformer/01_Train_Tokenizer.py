from tokenizers import BertWordPieceTokenizer
import os
from utils import load_data


X, Y = load_data("../json_data.json")

with open("./temp-1.txt", "w") as f:
    for x in X:
        f.write(x + "\n")

X, Y = load_data("../json_data_test.json")

with open("./temp-2.txt", "w") as f:
    for x in X:
        f.write(x + "\n")

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

tokenizer.train(files=["./temp-1.txt", "./temp-2.txt"], min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
os.remove("./temp-1.txt")
os.remove("./temp-2.txt")
if not os.path.exists("./bert-it"):
    os.mkdir("./bert-it")

tokenizer.save_model("./bert-it")