from fastapi import FastAPI
from utils.model_conf import get_model_info
from utils.qdrant_vector_database import QdrantVectorDatabase


app = FastAPI()
client = None

@app.on_event("startup")
async def startup_event():
    global client
    device, model, processor, tokenizer = get_model_info()
    client = QdrantVectorDatabase(device, model, tokenizer, processor)
    client.store_embeddings()


@app.post('/search/images')
async def search_text_to_image(query: str):
    global client
    images = client.text_to_images_search(query)
    return {'images': images}