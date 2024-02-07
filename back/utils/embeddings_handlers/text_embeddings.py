def get_prompt_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to list
    embedding_as_list = text_embeddings.squeeze().cpu().detach().numpy().tolist()
    return embedding_as_list
