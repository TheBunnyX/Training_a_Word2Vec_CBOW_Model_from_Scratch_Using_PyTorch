import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, Dataset

# Step 1: Preprocess the Text Data
def preprocess_text(filename):
    with open(filename, 'r') as file:
        text = file.read()
    words = text.lower().split()
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_idx = {word: idx for idx, word in enumerate(sorted_vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return words, word_to_idx, idx_to_word

# Step 2: Generate Training Samples
def generate_training_samples(words, word_to_idx, context_window_size):
    samples = []
    for i in range(context_window_size, len(words) - context_window_size):
        context = []
        for j in range(i - context_window_size, i + context_window_size + 1):
            if j != i:
                context.append(word_to_idx[words[j]])
        center_word = word_to_idx[words[i]]
        samples.append((context, center_word))
    return samples

# Step 3: Define the CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_mean = torch.mean(embedded, dim=1)
        scores = self.linear(embedded_mean)
        return scores

# Step 4: Train the CBOW Model
def train_cbow_model(words, word_to_idx, idx_to_word, vocab_size, embedding_dim, context_window_size, num_epochs, batch_size, learning_rate):
    dataset = CBOWDataset(words, word_to_idx, context_window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CBOW(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for context, center_word in dataloader:
            optimizer.zero_grad()
            outputs = model(context)
            loss = criterion(outputs, center_word)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')

    return model

class CBOWDataset(Dataset):
    def __init__(self, words, word_to_idx, context_window_size):
        self.samples = generate_training_samples(words, word_to_idx, context_window_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, center_word = self.samples[idx]
        return torch.tensor(context), torch.tensor(center_word)

# Step 5: Save the Trained Word Vectors
def save_word_vectors(model, idx_to_word, filename):
    embeddings = model.embedding.weight.data.numpy()
    with open(filename, 'w') as file:
        for idx, word in idx_to_word.items():
            embedding = ' '.join(map(str, embeddings[idx]))
            file.write(f'{word} {embedding}\n')

class Word2VecDataset(Dataset):
    def __init__(self, words, word_to_idx, context_window_size):
        self.samples = generate_training_samples(words, word_to_idx, context_window_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center_word, context_word = self.samples[idx]
        return torch.tensor(center_word), torch.tensor(context_word)

# Example usage:
filename = 'data.txt'
words, word_to_idx, idx_to_word = preprocess_text(filename)
vocab_size = len(word_to_idx)
embedding_dim = 100
context_window_size = 4
num_epochs = 1000
batch_size = 64
learning_rate = 0.01

model = train_cbow_model(words, word_to_idx, idx_to_word, vocab_size, embedding_dim, context_window_size, num_epochs, batch_size, learning_rate)
save_word_vectors(model, idx_to_word, 'word_vectors.txt')
