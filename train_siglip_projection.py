import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
from tqdm import tqdm

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # Apply layer normalization before projection
        x = F.layer_norm(x, x.size()[1:])
        x = self.linear(x)
        return x

def siglip_loss(image_embeddings, text_embeddings, temperature=0.07):
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    
    # Apply layer normalization before normalizing
    image_embeddings = F.layer_norm(image_embeddings, image_embeddings.size()[1:])
    text_embeddings = F.layer_norm(text_embeddings, text_embeddings.size()[1:])
    
    # Handle any remaining infinite values
    image_embeddings = torch.nan_to_num(image_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    text_embeddings = torch.nan_to_num(text_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize embeddings with better numerical stability
    image_embeddings = F.normalize(image_embeddings, dim=-1, eps=eps)
    text_embeddings = F.normalize(text_embeddings, dim=-1, eps=eps)

    # Compute pairwise similarities with better numerical stability
    logits = torch.matmul(image_embeddings, text_embeddings.transpose(0, 1))
    logits = logits / temperature
    
    # Clamp values to prevent overflow
    logits = torch.clamp(logits, min=-30.0, max=30.0)

    # Ground truth: 1.0 for matching pairs (diagonal), 0.0 for all others
    batch_size = logits.size(0)
    targets = torch.eye(batch_size).to(logits.device)

    # Apply binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Handle any NaN values
    if torch.isnan(loss).any():
        loss = torch.where(torch.isnan(loss), torch.tensor(10.0).to(loss.device), loss)
    
    return loss.mean()

def train_projection_layer(num_images=100, batch_size=32, num_epochs=50, learning_rate=1e-4, embedding_dim=2560):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR-10 dataset with minimal preprocessing
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),  # This scales to [0,1]
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, range(num_images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load models
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    siglip_model.eval()  # Freeze the model
    
    # Load Phi-2 model and tokenizer
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        device_map="auto",
        trust_remote_code=True
    )
    phi_model.eval()  # Freeze the model
    
    # Set up padding token
    if phi_tokenizer.pad_token is None:
        phi_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Use the EOS token as the pad token if needed
        if phi_tokenizer.pad_token is None:
            phi_tokenizer.pad_token = phi_tokenizer.eos_token
    
    # Print model configurations for debugging
    print("SigLIP config:", siglip_model.config)
    
    # Get embedding dimensions
    siglip_dim = siglip_model.config.vision_config.hidden_size  # SigLIP vision encoder hidden size
    
    print(f"SigLIP dimension: {siglip_dim}")
    print(f"Target embedding dimension: {embedding_dim}")
    
    # Initialize projection layer to project to a fixed embedding dimension
    projection_layer = LinearProjection(siglip_dim, embedding_dim).to(device)
    optimizer = torch.optim.Adam(projection_layer.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            # Convert images to PIL format for the processor
            images = [transforms.ToPILImage()(img) for img in images]
            
            # Get image embeddings from SigLIP
            with torch.no_grad():
                # Process images properly for SigLIP
                inputs = siglip_processor(images=images, return_tensors="pt").to(device)
                vision_outputs = siglip_model.vision_model(pixel_values=inputs.pixel_values)
                siglip_embeddings = vision_outputs.pooler_output  # Get pooled output from vision model
                
                # Apply layer normalization to SigLIP embeddings
                siglip_embeddings = F.layer_norm(siglip_embeddings, siglip_embeddings.size()[1:])
            
            # Project SigLIP embeddings
            projected_embeddings = projection_layer(siglip_embeddings)
            
            batch_loss = 0
            valid_lines = 0
            
            # Process each line from corresponding text files
            for line_idx in range(5):  # 5 lines per text file
                text_embeddings = []
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_images)
                
                try:
                    for img_idx in range(start_idx, end_idx):
                        with open(f"qa_outputs_extr/image_{img_idx}.txt", 'r') as f:
                            lines = f.readlines()
                            text = lines[line_idx].strip()
                            
                            # Get token embeddings directly from Phi-2's embedding layer
                            inputs = phi_tokenizer(text, return_tensors="pt", padding=True)
                            # Ensure input_ids are Long tensors and on the correct device
                            inputs.input_ids = inputs.input_ids.long().to(device)
                            with torch.no_grad():
                                # Get embeddings directly from the embedding layer
                                token_embeddings = phi_model.get_input_embeddings()(inputs.input_ids)
                                # Average the token embeddings (across sequence length)
                                text_embedding = token_embeddings.mean(dim=1)  # Shape will be [1, 2560]
                                text_embeddings.append(text_embedding)
                    
                    text_embeddings = torch.cat(text_embeddings, dim=0)  # Shape will be [batch_size, 2560]
                    
                    # Calculate loss for this line
                    line_loss = siglip_loss(projected_embeddings, text_embeddings)
                    
                    if not torch.isnan(line_loss) and line_loss > 0:
                        batch_loss += line_loss
                        valid_lines += 1
                
                except Exception as e:
                    print(f"Error processing line {line_idx} in batch {batch_idx}: {str(e)}")
                    continue
            
            # Only process batch if we have valid lines
            if valid_lines > 0:
                # Average loss over the valid lines
                batch_loss = batch_loss / valid_lines
                
                # Backpropagation
                optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += batch_loss.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No valid batches!")
    
    return projection_layer

if __name__ == "__main__":
    # Train the projection layer
    projection_layer = train_projection_layer(num_images=100)
    
    # Save the trained projection layer
    torch.save(projection_layer.state_dict(), 'trained_projection_layer.pth') 