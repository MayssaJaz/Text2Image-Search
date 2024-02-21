from PIL import Image
import numpy as np
import os


def save_batch_embeddings(embeddings_folder, embeddings, batch_num):
    """
    Save batch embeddings to a numpy file.

    Parameters:
        embeddings_folder (str): Path to the folder where embeddings will be saved.
        embeddings (numpy.ndarray): Batch of embeddings to be saved.
        batch_num (int): Batch number identifier.
    """
    filename = f"{embeddings_folder}/batch_{batch_num}_embeddings.npy"
    np.save(filename, embeddings)


def get_all_images_embedding(images_paths, batch_size, device, model, processor):
    """
    Extract embeddings for all images in batches.

    Parameters:
        images_paths (list): List of paths to image files.
        batch_size (int): Size of each batch for processing.
        device (torch.device): Device for computation (e.g., 'cpu' or 'cuda').
        model (torch.nn.Module): Model for extracting image features.
        processor (transformers.Processor): Processor for text and image inputs.
    """
    num_images = len(images_paths)
    embeddings_folder = os.environ["EMBEDDINGS_FOLDER"]
    os.mkdir(embeddings_folder)
    for i in range(0, num_images, batch_size):
        batch_paths = images_paths[i:i+batch_size]
        batch_embeddings = []
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as img:
                image = img.convert("RGB")
            batch_images.append(image)

        batch_inputs = processor(
            text=None,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_embedding = model.get_image_features(**batch_inputs)
        batch_embedding = batch_embedding.cpu().detach().numpy()
        for emb in batch_embedding:
            batch_embeddings.append(emb.squeeze().tolist())
        save_batch_embeddings(
            embeddings_folder, batch_embeddings, i // batch_size + 1)
    return len(batch_embeddings[0])
