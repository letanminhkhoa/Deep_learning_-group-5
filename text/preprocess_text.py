import re
import torch
import pandas as pd
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def load_and_clean_data():
    print("Downloading and loading 20 Newsgroups dataset...")
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    
    df_train = pd.DataFrame({'text': [clean_text(t) for t in train_data.data], 'label': train_data.target})
    df_train = df_train[df_train['text'] != '']
    
    df_test = pd.DataFrame({'text': [clean_text(t) for t in test_data.data], 'label': test_data.target})
    df_test = df_test[df_test['text'] != '']
    
    return df_train, df_test, train_data.target_names

class RNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = str(self.texts[idx]).split()
        encoded = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        if len(encoded) < self.max_length:
            encoded = encoded + [self.vocab["<PAD>"]] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_rnn_dataloaders(df_train, df_test, batch_size=32, max_length=256):
    all_words = ' '.join(df_train['text']).split()
    word_counts = Counter(all_words)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= 2:
            vocab[word] = idx
            idx += 1
            
    train_dataset = RNNDataset(df_train['text'], df_train['label'], vocab, max_length)
    test_dataset = RNNDataset(df_test['text'], df_test['label'], vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, vocab

def get_transformer_dataloaders(df_train, df_test, batch_size=32, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = TransformerDataset(df_train['text'], df_train['label'], tokenizer, max_length)
    test_dataset = TransformerDataset(df_test['text'], df_test['label'], tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, tokenizer