import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

# Load your trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = BertTokenizer.from_pretrained("./sentiment_model")

# Load the dataset
data = pd.read_csv('sampled_dataset.csv')  # Path to your dataset
data.dropna(subset=['text', 'emotions'], inplace=True)

# Encode emotions into numerical labels
label_encoder = LabelEncoder()
data['emotion_label'] = label_encoder.fit_transform(data['emotions'])

# Split dataset into test and train sets (use the same test set as in training)
test_texts = data['text'].tolist()
test_labels = data['emotion_label'].tolist()

# Function to make predictions using the trained model
def get_predictions(texts):
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()  # Get predicted class index
        predictions.append(predicted_label)
    return predictions

# Get model predictions for test data
predictions = get_predictions(test_texts)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)

# Emotion labels (your predefined labels)
emotion_labels = {
    0: "0", 1: "admiration", 2: "amusement", 3: "anger", 4: "annoyance", 5: "approval",
    6: "caring", 7: "confusion", 8: "curiosity", 9: "desire", 10: "disappointment",
    11: "disapproval", 12: "disgust", 13: "embarrassment", 14: "excitement", 15: "fear",
    16: "gratitude", 17: "grief", 18: "joy", 19: "love", 20: "nervousness", 21: "neutral",
    22: "optimism", 23: "pride", 24: "realization", 25: "relief", 26: "remorse",
    27: "sadness", 28: "surprise"
}

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels.values(), yticklabels=emotion_labels.values())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
