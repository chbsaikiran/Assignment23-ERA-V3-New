import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class QADataset(Dataset):
    def __init__(self, questions, num_images=100, max_length=512):
        self.questions = questions
        self.num_images = num_images
        self.max_length = max_length
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and processors
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        
        # Load CIFAR-10 test dataset
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.dataset = torch.utils.data.Subset(self.dataset, range(num_images))
        
        # Load models
        self.siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
        self.siglip_model.eval()
        
        # Load the projection layer
        self.projection_layer = LinearProjection(
            input_dim=1152,  # SigLIP hidden size
            output_dim=2560  # Phi-2's embedding size
        ).to(self.device)
        self.projection_layer.load_state_dict(torch.load('trained_projection_layer.pth', map_location=self.device))
        self.projection_layer.eval()
        
    def __len__(self):
        return len(self.dataset) * len(self.questions)
    
    def __getitem__(self, idx):
        image_idx = idx // len(self.questions)
        question_idx = idx % len(self.questions)
        
        # Get image and process it
        image, _ = self.dataset[image_idx]
        image = transforms.ToPILImage()(image)
        
        # Get image embeddings from SigLIP
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            vision_outputs = self.siglip_model.vision_model(pixel_values=inputs.pixel_values)
            siglip_embeddings = vision_outputs.pooler_output
            projected_embeddings = self.projection_layer(siglip_embeddings)
        
        # Get question and answer
        question = self.questions[question_idx]
        with open(f"qa_outputs_extr/image_{image_idx}_extr.txt", 'r') as f:
            lines = f.readlines()
            answer = lines[question_idx].strip()
        
        # Format input text with special tokens
        input_text = f"<|image|>{question}<|endoftext|>"
        target_text = f"{answer}<|endoftext|>"
        
        # Tokenize input and target
        model_inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize target text
        labels = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get the embeddings from input_ids
        input_ids = model_inputs.input_ids.squeeze(0)
        attention_mask = model_inputs.attention_mask.squeeze(0)
        labels = labels.input_ids.squeeze(0)
        
        # Move projected embeddings to CPU and detach
        image_embeddings = projected_embeddings.cpu().detach().squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_embeddings': image_embeddings
        }

class ImageConditionedPhi2(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.embed_dim = base_model.config.hidden_size
        
        # Add image embedding projection
        self.image_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, input_ids=None, attention_mask=None, image_embeddings=None, labels=None, inputs_embeds=None, **kwargs):
        # Handle either input_ids or inputs_embeds
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        if image_embeddings is not None:
            # Project and add image embeddings to the sequence
            image_proj = self.image_proj(image_embeddings)
            
            # Add image embeddings at the start of the sequence
            inputs_embeds = torch.cat([image_proj.unsqueeze(1), inputs_embeds], dim=1)
            
            # Extend attention mask for the added image token
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device),
                    attention_mask
                ], dim=1)
                
            # Adjust labels for the added image token
            if labels is not None:
                # Add ignore_index (-100) for the image token position
                labels = torch.cat([
                    torch.full((labels.shape[0], 1), -100, device=labels.device, dtype=labels.dtype),
                    labels
                ], dim=1)
        
        # Forward pass through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
        
    def generate(self, input_ids=None, attention_mask=None, image_embeddings=None, **kwargs):
        # Get base model embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        if image_embeddings is not None:
            # Project and add image embeddings
            image_proj = self.image_proj(image_embeddings)
            inputs_embeds = torch.cat([image_proj.unsqueeze(1), inputs_embeds], dim=1)
            
            # Extend attention mask for the added image token
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device),
                    attention_mask
                ], dim=1)
        
        # Call the base model's generate function with the modified inputs
        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # Get image embeddings from kwargs if available
        image_embeddings = kwargs.pop('image_embeddings', None)
        
        # Get base model embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        if image_embeddings is not None:
            # Project and add image embeddings
            image_proj = self.image_proj(image_embeddings)
            inputs_embeds = torch.cat([image_proj.unsqueeze(1), inputs_embeds], dim=1)
            
            # Extend attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device),
                    attention_mask
                ], dim=1)
        
        model_inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Add image embeddings to model inputs for future use
        if image_embeddings is not None:
            model_inputs['image_embeddings'] = image_embeddings
        
        return model_inputs
        
    def get_encoder(self):
        return self.base_model.get_encoder()

    def get_decoder(self):
        return self.base_model.get_decoder()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

def create_qlora_model():
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model with quantization config
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create image-conditioned model
    model = ImageConditionedPhi2(base_model)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h",],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model

def train(num_images=100):
    # Define questions
    questions = [
        "Give a description of the image?",
        "How does the main object in the image look like?",
        "How can the main object in the image be useful to humans?",
        "What is the color of the main object in the image?",
        "Describe the setting of the image?"
    ]
    
    # Create dataset
    dataset = QADataset(questions, num_images=num_images)
    
    # Create model
    model = create_qlora_model()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./phi2_qlora_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x]),
            'image_embeddings': torch.stack([item['image_embeddings'] for item in x])
        }
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model("./phi2_qlora_final")

def inference(image, question, model_path="./phi2_qlora_final", is_path=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and process image
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    
    if is_path:
        image = Image.open(image)
        image = transform(image).unsqueeze(0)
    else:
        # If image is already a tensor, just resize it
        if not isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            image = transform(image)
        image = image.unsqueeze(0)
    
    # Initialize models and processors
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    siglip_model.eval()
    
    # Load the projection layer
    projection_layer = LinearProjection(
        input_dim=1152,  # SigLIP hidden size
        output_dim=2560  # Phi-2's embedding size
    ).to(device)
    projection_layer.load_state_dict(torch.load('trained_projection_layer.pth', map_location=device))
    projection_layer.eval()
    
    # Get image embeddings from SigLIP
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        vision_outputs = siglip_model.vision_model(pixel_values=inputs.pixel_values)
        siglip_embeddings = vision_outputs.pooler_output
        projected_embeddings = projection_layer(siglip_embeddings)
    
    # Load the base model first with generation config
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    base_model.generation_config = GenerationConfig.from_pretrained("microsoft/phi-2")
    base_model.generation_config.max_new_tokens = 50  # Shorter responses
    base_model.generation_config.min_new_tokens = 5   # Ensure some minimum response
    base_model.generation_config.num_beams = 5        # More focused beam search
    base_model.generation_config.temperature = 0.3    # Lower temperature = more focused
    base_model.generation_config.do_sample = False    # Disable sampling for more deterministic outputs
    base_model.generation_config.top_p = 0.9
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    # Create image-conditioned model
    model = ImageConditionedPhi2(base_model)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "image_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Load trained weights
    model = PeftModel.from_pretrained(
        model,
        model_path,
        torch_dtype=torch.float16,
        is_trainable=False
    ).to(device)
    model.eval()
    
    # Format input text with special tokens
    input_text = f"<|image|>{question}<|endoftext|>"
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            image_embeddings=projected_embeddings
        )
        
        # Get generated text
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer by removing the question
        answer = answer.replace(input_text, "").strip()
        
    return answer

if __name__ == "__main__":
    # Load CIFAR-10 test dataset
    train(num_images=100)
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Get the first image from test set
    image, label = test_dataset[0]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"True label: {class_names[label]}")
    
    # Display image dimensions
    print(f"Image shape: {image.shape}")
    
    # List of questions
    questions = [
        "Give a description of the image?",
        "How does the main object in the image look like?",
        "How can the main object in the image be useful to humans?",
        "What is the color of the main object in the image?",
        "Describe the setting of the image?"
    ]
    
    print("\nGenerating answers for the image...")
    for question in questions:
        answer = inference(image, question, is_path=False)
        print("\nQuestion:", question)
        print("Answer:", answer) 