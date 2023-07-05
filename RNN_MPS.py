#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:50:39 2023

@author: hbonen
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import time

start_time = time.time()

# Import the Cornell Movie Dialogs Corpus
corpus_path = '/Users/hbonen/Documents/chatbot/cornell_movie_quotes_corpus/moviequotes.memorable_nonmemorable_pairs_original.txt'

# Read the conversations from the dataset
conversations = []
with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as file:
    lines = file.readlines()
    conversation = []
    for line in lines:
        line = line.strip()
        if line:
            conversation.append(line.split('+++$+++')[-1].strip())
        else:
            if len(conversation) > 1:
                conversations.append(conversation)
            conversation = []

# Training data
input_sequences = []
output_sequences = []
for conversation in conversations:
    for i in range(len(conversation) - 1):
        input_sequences.append(conversation[i])
        output_sequences.append(conversation[i+1])

class ChatbotDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.tokenizer = None
        self.total_words = 0
        self._preprocess_data()

    def _preprocess_data(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(self.input_sequences)
        self.tokenizer.add_tokens(self.output_sequences)
        self.total_words = len(self.tokenizer)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        input_text = self.input_sequences[index]
        output_text = self.output_sequences[index]

        input_encoded = self.tokenizer.encode(input_text, add_special_tokens=True)
        output_encoded = self.tokenizer.encode(output_text, add_special_tokens=True)

        return torch.tensor(input_encoded), torch.tensor(output_encoded)

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output

hidden_size = 1
learning_rate = 0.001
batch_size = 1
num_epochs = 100

# Choose the device type ('cpu' or 'mps')
# device_type = 'mps'  # Set to 'cpu' or 'mps' as desired

# Create dataset and dataloader
dataset = ChatbotDataset(input_sequences, output_sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, loss function, and optimizer
# device = torch.device('cpu')  # Default device is CPU

# Load the entire model
model = torch.load('model.pth')
# Create an instance of the model and load the state dictionary
model = ChatbotModel(dataset.total_words, hidden_size, dataset.total_words)
model.load_state_dict(torch.load('model_state.pth'))


def generate_response(input_text):
    input_encoded = dataset.tokenizer.encode(input_text, add_special_tokens=True)
    input_tensor = torch.tensor(input_encoded).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    
    output = output.squeeze()
    
    if output.dim() == 1:
        output = output.unsqueeze(0)
    
    predicted_word_index = torch.argmax(output, dim=1)
    predicted_word = dataset.tokenizer.decode(predicted_word_index.tolist(), skip_special_tokens=True)

    return predicted_word

user_input = 'Can I take a coffee?'
response = generate_response(user_input)
print(response)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
