import numpy as np
from qdrant_client import models
import uuid


def load_batch_embeddings(batch_num, folder_path):
    """
    Load batch embeddings from a numpy file.

    Parameters:
        batch_num (int): Batch number identifier.
        folder_path (str): Path to the folder where embeddings are saved.

    Returns:
        numpy.ndarray: Batch embeddings loaded from the file.
    """
    filename = f"{folder_path}/batch_{batch_num}_embeddings.npy"
    return np.load(filename)


def upsert_batch_embeddings(client, collection_name, batch_embeddings, images_paths):
    """
    Upsert batch embeddings into a Qdrant collection.

    Parameters:
        client (qdrant_client.client.QdrantClient): Qdrant client for interacting with Qdrant.
        collection_name (str): Name of the collection where embeddings will be upserted.
        batch_embeddings (numpy.ndarray): Batch of embeddings to be upserted.
    """
    batch_size = len(batch_embeddings)
    batch_ids = [str(uuid.uuid4())
                 for i in range(batch_size)]  # Generating IDs for the batch
    paths_dict = [{'image_path': path} for path in images_paths]
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=batch_ids,
            payloads=paths_dict,
            vectors=batch_embeddings
        ),
    )
