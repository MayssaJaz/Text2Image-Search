from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from utils.embeddings_handlers.images_embeddings import get_all_images_embedding
from utils.embeddings_handlers.text_embeddings import get_prompt_text
from utils.batch_handler.batch_processing import load_batch_embeddings, upsert_batch_embeddings


class QdrantVectorDatabase:
    def __init__(self, device, model, tokenizer, processor):
        self.client = QdrantClient(os.environ['QDRANT_URL'])
        self.device = device
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def store_embeddings(self):
        """
        Stores all image embeddings inside the QDRANT vector database.
        This function retrieves embeddings for all images in the specified directory, 
        creates a new collection in the QDRANT database to store the embeddings, 
        and then inserts the embeddings along with their corresponding image paths 
        into the created collection."""
        # Get all images
        directory_path = os.environ['DATASET_PATH']
        list_files = os.listdir(directory_path)
        image_paths = [directory_path + '/' + path for path in list_files]
        print('Generating images embeddings...')
        embedding_size = get_all_images_embedding(
            image_paths, 64, self.device, self.model, self.processor)
        print('Got all embeddings!')
        # Create a new collection to store our images embeddings
        self.client.recreate_collection(
            collection_name=os.environ['COLLECTION_NAME'],
            vectors_config=VectorParams(
                size=embedding_size, distance=Distance.COSINE),
        )
        # Waiting message
        print("Storing images embeddings. Please wait...")
        # Upsert points (embeddings) into the created collection using batch processing
        embeddings_folder = os.environ["EMBEDDINGS_FOLDER"]
        list_files_embeddings = os.listdir(embeddings_folder)
        images_paths_batches = [list_files[i:i+64]
                                for i in range(0, len(list_files), 64)]
        # Loop over embeddings files and stocking their vectors
        for batch_num in range(1, len(list_files_embeddings) + 1):
            batch_embeddings = load_batch_embeddings(
                batch_num, embeddings_folder)
            upsert_batch_embeddings(
                self.client, os.environ['COLLECTION_NAME'], batch_embeddings, images_paths_batches[batch_num - 1])
        # Success message
        print("Embeddings stored successfully")

    def text_to_images_search(self, query):
        """
        Retrieve images similar to the text query based on their embeddings and cosine similarity.

        Args:
            query (str): The text query for finding similar images.

        Returns:
            list: A list of paths to the most similar images to the query."""
        # Retrieve the embedding of our text query and use it as a query vector to search for the most similar images vectors within our collection
        search_result = self.client.search(
            collection_name=os.environ['COLLECTION_NAME'],
            query_vector=get_prompt_text(query, self.model, self.tokenizer),
            limit=5
        )
        # Collect the paths of the most similar images to the query
        selected_images = []
        for result in search_result:
            selected_images.append(result.payload['image_path'])
        return selected_images
