from transformers import BertForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, BertTokenizer, BertConfig
from utils import CustomDataset, get_word_index, load_data
import numpy as np
from datasets import load_metric
from seqeval.metrics import classification_report
import pandas as pd
import re
import os

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

    token_labels = ["O", "TASK", "PROCESS", "MATERIAL"]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    net_results = {}
    for t in token_labels:
        net_results[t] = {}
        net_results[t]['precision'] = results[t]['precision']
        net_results[t]['recall'] = results[t]['recall']
        net_results[t]['f1-score'] = results[t]['f1']

    net_results['overall'] = {}
    net_results['overall']['precision'] = results["overall_precision"]
    net_results['overall']['recall'] = results["overall_recall"]
    net_results['overall']['f1-score'] = results["overall_f1"]
    net_results['overall']['accuracy'] = results["overall_accuracy"]

    results = classification_report(true_predictions, true_labels)

    print(results)

    return net_results


tokenizer = BertTokenizer.from_pretrained('./bert-it')
data_collator = DataCollatorForTokenClassification(tokenizer)
model = BertForTokenClassification.from_pretrained('./bert-it')
dataset = CustomDataset("./data/test-data.parquet")

training_args = TrainingArguments(
    output_dir="./bert-it",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=2,
    run_name="NER",
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics
)
trainer.evaluate()
