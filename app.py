# sentiment_app.py
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the model and tokenizer
model_path = "./sentiment_model"  # Path to your saved model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Define label-to-emotion mapping (replace with your actual labels)
emotion_labels = {
    0: "0", 1: "admiration", 2: "amusement", 3: "anger", 4: "annoyance", 5: "approval",
    6: "caring", 7: "confusion", 8: "curiosity", 9: "desire", 10: "disappointment",
    11: "disapproval", 12: "disgust", 13: "embarrassment", 14: "excitement", 15: "fear",
    16: "gratitude", 17: "grief", 18: "joy", 19: "love", 20: "nervousness", 21: "neutral",
    22: "optimism", 23: "pride", 24: "realization", 25: "relief", 26: "remorse",
    27: "sadness", 28: "surprise"
}

# Reverse mapping for labels to emotions
label_to_emotion = {v: k for k, v in emotion_labels.items()}

# Define a prediction function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    emotion = emotion_labels.get(prediction, "Unknown Emotion")  # Convert label to emotion name
    return emotion

# Function to evaluate model and calculate accuracy and confusion matrix
def evaluate_model(test_texts, test_labels):
    predictions, true_labels = [], []

    for text, label in zip(test_texts, test_labels):
        predicted_label = label_to_emotion[predict_emotion(text)]
        predictions.append(predicted_label)
        true_labels.append(label)

    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    return accuracy, conf_matrix

# Streamlit UI
st.title("Text Emotion Analysis with BERT")
st.write("Enter a sentence to predict its emotional sentiment.")

# Text input for user
user_input = st.text_input("Your Text:", "")

# Predict button
if st.button("Predict Emotion"):
    if user_input:
        emotion = predict_emotion(user_input)
        st.write(f"Predicted Emotion: {emotion}")
    else:
        st.write("Please enter some text for prediction.")


# Load test data for evaluation (replace with actual test data)
# Ensure test_texts and test_labels contain your test text and label data
test_texts = ["sample text 1", "sample text 2"]  # Replace with actual test texts
test_labels = [0, 1]  # Replace with actual test labels

