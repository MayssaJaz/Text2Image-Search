def get_prompt_text(text, model, tokenizer):
    """
    Generates text embeddings for a given prompt text using the specified CLIP model.

    This function tokenizes the input text, generates embeddings using the CLIP model,
    and returns the generated embeddings as a list.

    Args:
        text (str): The prompt text for which embeddings are to be generated.
        model (CLIPModel): The CLIP model used for generating embeddings.
        tokenizer (CLIPTokenizer): The tokenizer associated with the CLIP model.

    Returns:
        list: A list containing the embeddings generated for the input text.
    """
    # Tokenize our text
    inputs = tokenizer(text.lower(), return_tensors="pt")
    # Generate embeddings
    text_embeddings = model.get_text_features(**inputs)
    # Convert the embeddings to list
    embedding_as_list = text_embeddings.squeeze().cpu().detach().numpy().tolist()
    return embedding_as_list
