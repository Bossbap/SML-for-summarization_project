import os
import torch
from torch.utils.data import Dataset

# Load and split dataset
data_path = "data/toy_dataset"
files = sorted(os.listdir(data_path))

cleaned_texts = []
summaries = []

# Assume first half are cleaned texts and second half are summaries
midpoint = len(files) // 2
for file in files[:midpoint]:
    with open(os.path.join(data_path, file), "r") as f:
        cleaned_texts.append(f.read())
for file in files[midpoint:]:
    with open(os.path.join(data_path, file), "r") as f:
        summaries.append(f.read())

data_pairs = list(zip(cleaned_texts, summaries))

# Define dataset class
class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig, summ = self.data[idx]
        return orig, summ

# Create dataset object
dataset = SummarizationDataset(data_pairs)

# Save dataset object
os.makedirs("data", exist_ok=True)
torch.save(dataset, "data/toy_dataset.pt")

print("Toy dataset loaded and saved successfully.")