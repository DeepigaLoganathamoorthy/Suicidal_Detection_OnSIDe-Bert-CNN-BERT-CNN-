
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#hyperparameters
max_words = 256
max_sequence_length = 128

# Pad sequences
X_train_pad = pad_sequences(train_texts, maxlen=max_sequence_length, padding='post')
X_val_pad = pad_sequences(val_texts, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(test_texts, maxlen=max_sequence_length, padding='post')

# Build the CNN model
model = Sequential()
model.add(Embedding(max_words, 50, input_length=max_sequence_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


optimizer = Adam(lr=0.00002)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


early_stopping = EarlyStopping(patience=15, monitor='val_loss')


history = model.fit(X_train_pad, train_labels, validation_data=(X_val_pad, val_labels), epochs=10, batch_size=32, callbacks=[early_stopping])

# Evaluate on validation set
val_pred = model.predict_classes(X_val_pad)
val_accuracy = accuracy_score(val_labels, val_pred)
val_precision = precision_score(val_labels, val_pred)
val_recall = recall_score(val_labels, val_pred)
val_f1 = f1_score(val_labels, val_pred)

print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Validation Precision: {val_precision:.4f}')
print(f'Validation Recall: {val_recall:.4f}')
print(f'Validation F1 Score: {val_f1:.4f}')

# Confusion matrix for validation set
val_conf_matrix = confusion_matrix(val_labels, val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(val_conf_matrix, annot=True, cmap='Greens', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Validation Set')
plt.show()

# Evaluate on test set
test_pred = model.predict_classes(X_test_pad)
test_accuracy = accuracy_score(test_labels, test_pred)
test_precision = precision_score(test_labels, test_pred)
test_recall = recall_score(test_labels, test_pred)
test_f1 = f1_score(test_labels, test_pred)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')

# Confusion matrix for test set
test_conf_matrix = confusion_matrix(test_labels, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, cmap='Greens', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Set')
plt.show()
