import json
import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
from tqdm import tqdm 
from transformers import AutoModel, BertTokenizerFast
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.optim import AdamW, lr_scheduler
from torchinfo import summary
from transformers import pipeline

pipe = pipeline("fill-mask", model="emilyalsentzer/Bio_ClinicalBERT")

# Load intents data
def load_intents(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Intent file not found.")
        return None

data = load_intents("/Users/zhen-meitan/Desktop/Personal/Uni/Projektstudium/BertModel/intents.json")

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

df = load_data('BertModel/dataset.csv')

# oversampling
def balance_dataset(df, target_column):
    majority_class_label = df[target_column].mode()[0]

    minority_classes = [df[df[target_column] == cls] for cls in df[target_column].unique() if cls != majority_class_label]

    resampled_data = [resample(minority, replace=True, n_samples=len(df[df[target_column] == majority_class_label]), random_state=42) for minority in minority_classes]
    df = pd.concat([df[df[target_column] == majority_class_label]] + resampled_data)
    return df

df = balance_dataset(df, 'diagnosis')

# Convert labels to encodings
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Create the train dataset
train_text, train_labels = df['symptoms'], df['diagnosis']

# Load BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")  
# Define maximum sequence length
max_seq_len = 128

# Tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# Prepare tensors for training
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# Create DataLoader
batch_size = 16
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# BERT Architecture for fine-tuning
class BERT_Arch(nn.Module):
    def __init__(self, bert, num_classes):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Initialize the model with the number of classes
num_classes = df['diagnosis'].nunique()
model = BERT_Arch(bert, num_classes)
device = torch.device('mps')  
model = model.to(device)

# Model summary
summary(model)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-03)  
class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_wts, dtype=torch.float).to(device)
print("Class weights:", weights)

cross_entropy = nn.NLLLoss(weight=weights)

# Function to save model checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir="checkpoint"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt")
    print(f"Checkpoint for epoch {epoch+1} saved.")

# Function to train the model
epochs = 5
train_losses = []

lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def train():
    model.train()
    total_loss = 0
    total_preds = []

    # Initialize tqdm for the progress bar in training loop
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", unit="batch")

    for step, batch in pbar:
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

        # Update tqdm progress bar
        pbar.set_postfix(loss=total_loss / (step + 1), refresh=True)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Training loop with progress bar and time remaining
for epoch in range(epochs):
    print(f'\nEpoch {epoch+1} / {epochs}')
    train_loss, _ = train()
    train_losses.append(train_loss)

    # Optionally, save checkpoint after each epoch
    save_checkpoint(epoch, model, optimizer, train_loss)

    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.3f}")

print(f'\nFinal Training Loss: {train_loss:.3f}')
import torch
from transformers import BertTokenizerFast
import numpy as np

# Load the trained model checkpoint (if needed)
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Epoch {epoch}, Loss {loss}")
    return model, optimizer

def predict(model, tokenizer, text, device, max_seq_len=128):
    # Prepare the input text for prediction
    tokens = tokenizer(
        text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_tensors="pt"  # Use pytorch tensors
    )

    # Move input tensors to the same device as the model
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Turn off gradient calculations (no backpropagation)
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)  # Get the class with the highest score

    return predictions.item()  # Return the predicted class label


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert, num_classes=df['diagnosis'].nunique())
model.to(device) 


model.eval()
model.to(device)  

print("Welcome to the Medical Diagnosis Chatbot!")
print("Please describe your symptoms, or type 'exit' to end the chat.")

while True:
    input_text = input("You: ")
    
    if input_text.lower() == 'exit':
        print("Goodbye! Take care.")
        break
    
    predicted_label = predict(model, tokenizer, input_text, device)
    predicted_class = le.inverse_transform([predicted_label])
    
    print(f"Bot: Based on your symptoms, the predicted diagnosis is: {predicted_class[0]}")

