#!/usr/bin/env python
# coding: utf-8

# # Language Modeling
# ## Kiara LaRocca | klarocca@ucsc.edu
# ________________________________________
# ## NLP 243 | HW 3
# ### 25 November 2024

# In[ ]:


import argparse
import sys
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import csv

# Check if the script is executed with exactly one command-line argument
if len(sys.argv) != 2:
    print("Usage: python run.py <output>")
    sys.exit(1)

# Argument parsing
parser = argparse.ArgumentParser(description="Run Transformer and save results")
parser.add_argument("output", type=str, help="Output file name for results")
args = parser.parse_args()

# Load the Penn Treebank Dataset
print("Loading dataset...")
ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
train_sentences = ptb['train']['sentence']
val_sentences = ptb['validation']['sentence']
test_sentences = ptb['test']['sentence']

# Tokenize Sentences
class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.current_id = len(self.vocab)
        self.vocab_size = vocab_size

    def encode(self, sentence, add_special_tokens=True, max_length=128):
        tokens = sentence.split()
        if add_special_tokens:
            tokens = ["<s>"] + tokens + ["</s>"]
        token_ids = [self.get_token_id(token) for token in tokens[:max_length]]
        return token_ids

    def get_token_id(self, token):
        if token not in self.vocab:
            if len(self.vocab) < self.vocab_size:
                self.vocab[token] = self.current_id
                self.current_id += 1
            else:
                return self.vocab["<unk>"]
        return self.vocab[token]

    @property
    def pad_token_id(self):
        return self.vocab["<pad>"]

    @property
    def actual_vocab_size(self):
        return len(self.vocab)

tokenizer = SimpleTokenizer()

def tokenize_datasets(datasets):
    tokenized = {}
    for split_name, sentences in datasets.items():
        print(f"Starting tokenization for {split_name} set...")
        tokenized[split_name] = [tokenizer.encode(sentence) for sentence in sentences]
    return tokenized

datasets = {"train": train_sentences, "validation": val_sentences, "test": test_sentences}
tokenized_data = tokenize_datasets(datasets)
train_tokens, val_tokens, test_tokens = tokenized_data["train"], tokenized_data["validation"], tokenized_data["test"]

# Define the Dataset Class
class PTBDataset(Dataset):
    def __init__(self, tokenized_sentences):
        self.data = tokenized_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return torch.tensor(tokens[:-1], dtype=torch.long), torch.tensor(tokens[1:], dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    return inputs_padded, targets_padded

batch_size = 32
train_loader = DataLoader(PTBDataset(train_tokens), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Transformer Components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        return self.out(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, max_seq_len, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.actual_vocab_size
d_model = 512
num_heads = 4
num_layers = 6
ff_hidden_dim = 2048
max_seq_len = 128
dropout = 0.5
model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, max_seq_len, dropout).to(device)

# Training and Validation Logic
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

num_epochs = 5
gradient_clip = 1.0
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}...")
    
    # Training Loop
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Mask for the transformer
        seq_len = inputs.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

        # Forward pass
        logits = model(inputs, mask)
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)

        # Compute loss
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation Loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            seq_len = inputs.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

            logits = model(inputs, mask)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

# Perplexity Calculation
def calculate_perplexity(model, tokenized_sentence):
    model.eval()
    device = next(model.parameters()).device
    input_seq = torch.tensor(tokenized_sentence[:-1], device=device).unsqueeze(0)
    target_seq = torch.tensor(tokenized_sentence[1:], device=device).unsqueeze(0)
    seq_len = input_seq.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_seq, mask)
        logits = logits.view(-1, vocab_size)
        target_seq = target_seq.view(-1)
        loss = criterion(logits, target_seq)
        perplexity = torch.exp(loss)
    return perplexity.item()

# Calculate Perplexity
results = []
for i, sentence in enumerate(test_tokens):
    ppl = calculate_perplexity(model, sentence)
    results.append((i, ppl))

# Save Results to CSV
with open(args.output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "ppl"])
    writer.writerows(results)

print(f"Results saved to {args.output}")


# In[ ]:




