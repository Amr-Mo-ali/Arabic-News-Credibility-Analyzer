import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv('data/processed/arabic_fake_news_dataset.csv')
# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
model = AutoModelForSequenceClassification.from_pretrained('aubmindlab/bert-base-arabertv2', num_labels=2)
# Tokenize the data and create a Dataset object
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Create Dataset objects
# We need to convert the labels to a format that can be used by the Trainer
train_encodings['labels'] = train_labels
test_encodings['labels'] = test_labels
# Convert the encodings to a Dataset object
# add the labels to the encodings
train_dataset = Dataset.from_dict(train_encodings)
test_dataset = Dataset.from_dict(test_encodings)

# Set the format for the datasets
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100  
)

def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score for the evaluation predictions.

    Parameters:
    eval_pred (tuple): A tuple containing the predictions and true labels.

    Returns:
    dict: A dictionary containing the accuracy and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Create the Trainer object
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,          # evaluation dataset
    compute_metrics=compute_metrics      # the function that will be used to compute metrics at evaluation
)