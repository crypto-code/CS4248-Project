from transformers import BertForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, BertTokenizer, BertConfig
import warnings
warnings.filterwarnings("ignore")
from utils import CustomDataset
import numpy as np
from datasets import load_metric
import torch
import os
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

torch.cuda.empty_cache()

dataset = CustomDataset("./data/train-data.parquet")

train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=1801)

id2label = {
    0: "I-O",
    1: "I-TASK",
    2: "I-PROCESS",
    3: "I-MATERIAL"
}

metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(true_predictions, true_labels)

    print(results)

    return {}


tokenizer = BertTokenizer.from_pretrained('./bert-it')
data_collator = DataCollatorForTokenClassification(tokenizer)
config = BertConfig.from_json_file("./model_config.json")
model = BertForTokenClassification(config)
if os.path.exists("./bert-it/pytorch_model.bin"):
    model = BertForTokenClassification.from_pretrained('./bert-it')

training_args = TrainingArguments(
    output_dir="./bert-it",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=25,
    weight_decay=0.01,
    logging_steps=500,
    run_name="NER",
    evaluation_strategy="steps"
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
model.save_pretrained('./bert-it')
