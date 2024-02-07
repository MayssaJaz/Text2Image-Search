from PIL import Image


def get_single_image_embedding(image_path, device, model, processor):
    """
    Generates embeddings for a single image.

    This function takes the path to an image file, loads and processes the image, 
    generates embeddings using the specified CLIP model, converts the embeddings 
    to a list format then returns the list of embeddings.

    Args:
        image_path (str): The file path to the image.
        device (str): The device ('cuda' if available, otherwise 'cpu').
        model (CLIPModel): The CLIP model used for generating embeddings.
        processor (CLIPProcessor): The processor associated with the CLIP model.

    Returns:
        list: A list containing the embeddings generated for the input image.
    """
    # Generate embeddings for the image with image_path
    image = processor(
        text=None,
        images=Image.open(image_path).convert("RGB"),
        return_tensors="pt"
    )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # Convert the embeddings to list
    embedding_as_list = embedding.squeeze().cpu().detach().numpy().tolist()
    return embedding_as_list


def get_all_images_embedding(images_paths, device, model, processor):
    # Loop over all images paths and generate their embeddings
    images_embeddings = [get_single_image_embedding(
        path, device, model, processor) for path in images_paths]
    return images_embeddings
