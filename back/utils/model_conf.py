from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import torch
import os


def get_model_info():
    """
    Retrieves information about the model.
    This function sets the device based on availability, defines the model ID 
    obtained from the environment variable MODEL_ID, loads the CLIP model onto the specified 
    device, and retrieves the processor and tokenizer associated with the model.

    Returns:
        device (str): The device ('cuda' if available, otherwise 'cpu').
        model (CLIPModel): The CLIP model loaded onto the specified device.
        processor (CLIPProcessor): The processor associated with the CLIP model.
        tokenizer (CLIPTokenizer): The tokenizer associated with the CLIP model.
    """
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