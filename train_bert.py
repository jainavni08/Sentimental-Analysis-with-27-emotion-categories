import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load dataset
data = pd.read_csv('sampled_dataset.csv') 
data.dropna(subset=['text', 'emotions'], inplace=True)

# Encode labels
label_encoder = LabelEncoder()
data['emotion_label'] = label_encoder.fit_transform(data['emotions'])
num_emotions = data['emotion_label'].nunique()

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'].tolist(), data['emotion_label'].tolist(), test_size=0.2, random_state=42
)

# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        encoding['labels'] = torch.tensor(labels)
        return encoding

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_emotions)

# Prepare datasets
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Save the model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# Function to make predictions
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    emotion = label_encoder.inverse_transform([prediction])
    return emotion[0]

# Test the prediction function
test_text = "should i wear blck or white tshirt?"
print("Predicted Emotion:", predict_emotion(test_text))
