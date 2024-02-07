from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import torch
import os

def get_model_info():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the model ID
    model_ID = os.environ['MODEL_ID']
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return device, model, processor, tokenizer
