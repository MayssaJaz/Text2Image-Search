from fastapi import FastAPI
from utils.model_conf import get_model_info
from utils.qdrant_vector_database import QdrantVectorDatabase
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = None
# Mount the directory containing the images to serve them statically
app.mount("/images", StaticFiles(directory="dataset"), name="dataset")


@app.on_event("startup")
async def startup_event():
    global client
    device, model, processor, tokenizer = get_model_info()
    client = QdrantVectorDatabase(device, model, tokenizer, processor)
    client.store_embeddings()


@app.post('/search/images')
async def search_text_to_image(query: dict):
    global client
    images = client.text_to_images_search(query.get('query', ''))
    return {'images': images}
