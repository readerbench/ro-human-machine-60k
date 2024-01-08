'''
Transformer-Based Detector
Multiclass text classification using readerbench/robert-base model
Dataset can be found in /dataset dir
The machine_generated_dataset.xlsx file contains human and machine generated texts from 5 domains
For an easier classification, we considered iteratively texts from each domain 
The implementation displays classification report and confusion matrix
'''

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

data = pd.read_csv('mgt_dataset_path') # check dataset dir for repro
data.dropna(inplace=True)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Extract texts and labels for each class
texts = []
labels = []

for label in data.columns[1:]:
    class_texts = data[label].tolist()
    texts.extend(class_texts)
    labels.extend([label] * len(class_texts))

# Create label mapping
label_map = {label: idx for idx, label in enumerate(data.columns[1:])}

# Map labels to integer values for training
int_labels = [label_map[label] for label in labels]

# Split the dataset into train and test sets with stratified labels
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, int_labels, test_size=0.2, random_state=42, stratify=int_labels)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("readerbench/robert-base")
model = AutoModelForSequenceClassification.from_pretrained("readerbench/robert-base", num_labels=len(data.columns[1:]))

# Tokenize the input texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create dataset objects
train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Create a Trainer instance
training_args = TrainingArguments(
    output_dir="./robert-base-classifier",
    evaluation_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    # add early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # add gradient accumulation
    gradient_accumulation_steps=2,
    save_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.predict(test_dataset)

# Get predicted labels
predicted_labels = results.predictions.argmax(axis=1)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average="weighted")
recall = recall_score(test_labels, predicted_labels, average="weighted")
f1 = f1_score(test_labels, predicted_labels, average="weighted")

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Weighted F1-score: {f1:.4f}")

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Original_Text': test_texts,
    'True_Label': [data.columns[1:][label_id] for label_id in test_labels],
    'Predicted_Label': [data.columns[1:][label_id] for label_id in predicted_labels]
})

#results_df.to_csv('multiclass.csv', index=False, encoding='utf-8-sig')

conf_matrix = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

