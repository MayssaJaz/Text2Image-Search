from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from utils.embeddings_handlers.images_embeddings import get_all_images_embedding
from utils.embeddings_handlers.text_embeddings import get_prompt_text


class QdrantVectorDatabase:
    def __init__(self, device, model, tokenizer, processor):
        self.client = QdrantClient(os.environ['QDRANT_URL'])
        self.device = device
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def store_embeddings(self):
        directory_path = os.environ['DATASET_PATH']
        list_files = os.listdir(directory_path)
        image_paths = [directory_path + '/' + path for path in list_files]
        print('Generating images embeddings...')
        images_embeddings = get_all_images_embedding(
            image_paths, self.device, self.model, self.processor)
        print('Got all embeddings!')
        # Create a new collection to store our images embeddings
        self.client.recreate_collection(
            collection_name="mayssa",
            vectors_config=VectorParams(
                size=len(images_embeddings[0]), distance=Distance.COSINE),
        )
        # Waiting message
        print("Storing images embeddings. Please wait...")
        # Upsert points (embeddings) into the created collection
        for embedding, path in zip(images_embeddings, image_paths):
            self.client.upsert(
                collection_name="mayssa",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={"image_path": path}
                    )
                ]
            )
        # Success message
        print("Embeddings stored successfully")

    def text_to_images_search(self, query):
        # Retrieve the embedding of our text query and use it as a query vector to search for the most similar images vectors within our collection
        search_result = self.client.search(
            collection_name="mayssa",
            query_vector=get_prompt_text(query, self.model, self.tokenizer),
            limit=5
        )
        # Collect the paths of the most similar images to the query
        selected_images = []
        for result in search_result:
            selected_images.append(result.payload['image_path'])
        return selected_images
