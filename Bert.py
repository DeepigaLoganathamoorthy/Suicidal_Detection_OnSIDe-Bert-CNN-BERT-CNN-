
pip install transformers

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

texts = df['text'].tolist()
labels = df['class'].tolist()
label_mapping = {'suicide': 0, 'non-suicide': 1}
labels = [label_mapping[label] for label in labels]

#Load pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
pretrained_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_hidden_states=False)

#Tokenize
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
train_texts, test_texts, train_labels, test_labels = train_test_split(encoded_texts.input_ids, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.375, random_state=42)

train_masks = encoded_texts['attention_mask'][train_texts]
val_masks = encoded_texts['attention_mask'][val_texts]
test_masks = encoded_texts['attention_mask'][test_texts]

train_dataset = TensorDataset(train_texts, train_masks, torch.tensor(train_labels))
val_dataset = TensorDataset(val_texts, val_masks, torch.tensor(val_labels))
test_dataset = TensorDataset(test_texts, test_masks, torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(pretrained_model.parameters(), lr=2e-5)

# Fine-tuning the BERT model
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)

for epoch in range(num_epochs):
    # Training loop
    pretrained_model.train()
    for inputs, masks, labels in train_loader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = pretrained_model(inputs, attention_mask=masks)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    pretrained_model.eval()
    val_true_labels = []
    val_predicted_labels = []
    with torch.no_grad():
        for inputs, masks, labels in val_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = pretrained_model(inputs, attention_mask=masks)[0]
            val_predicted_labels.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_true_labels, val_predicted_labels)
    val_precision = precision_score(val_true_labels, val_predicted_labels)
    val_recall = recall_score(val_true_labels, val_predicted_labels)
    val_f1 = f1_score(val_true_labels, val_predicted_labels)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

print("Finished fine-tuning")

# Evaluation on test set
pretrained_model.eval()
test_true_labels = []
test_predicted_labels = []

with torch.no_grad():
    for inputs, masks, labels in test_loader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        outputs = pretrained_model(inputs, attention_mask=masks)[0]
        test_predicted_labels.extend(torch.argmax(outputs, axis=1).cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
test_precision = precision_score(test_true_labels, test_predicted_labels)
test_recall = recall_score(test_true_labels, test_predicted_labels)
test_f1 = f1_score(test_true_labels, test_predicted_labels)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Confusion matrix validation set
val_conf_matrix = confusion_matrix(val_true_labels, val_predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(val_conf_matrix, annot=True, cmap='Greens', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Validation Set')
plt.show()

val_report = classification_report(val_true_labels, val_predicted_labels)
print("Classification Report for Validation Set:")
print(val_report)

# Confusion matrix test set
test_conf_matrix = confusion_matrix(test_true_labels, test_predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, cmap='Greens', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Set')
plt.show()

test_report = classification_report(test_true_labels, test_predicted_labels)
print("Classification Report for Test Set:")
print(test_report)
