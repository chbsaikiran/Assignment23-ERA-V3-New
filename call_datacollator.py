import torch
from torch.utils.data import DataLoader
from train_phi2_qlora import QADataset

# Sample 5 questions for each of 2 images (5 x 2 = 10 samples)
questions = [
    "Give a description of the image?",
    "How does the main object in the image look like?",
    "How can the main object in the image be useful to humans?",
    "What is the color of the main object in the image?",
    "Describe the setting of the image?"
]

# Create dataset with just 2 images = 10 samples (2 images × 5 questions)
dataset = QADataset(questions, num_images=2)

# Define the same data collator used in Trainer
data_collator = lambda x: {
    'input_ids': torch.stack([item['input_ids'] for item in x]),
    'attention_mask': torch.stack([item['attention_mask'] for item in x]),
    'labels': torch.stack([item['labels'] for item in x]),
    'image_embeddings': torch.stack([item['image_embeddings'] for item in x])
}

# Create DataLoader with batch_size = 10
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

# Get a batch of data
batch = next(iter(dataloader))

# Print keys and shapes
for k, v in batch.items():
    print(f"{k}: value = {v}, shape = {v.shape}")
