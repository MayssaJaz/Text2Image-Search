# Text2Image-Search

## Purpose of the Project
This project aims create a search engine designed specifically for retrieving images from a dataset using textual queries, a process commonly known as text-to-image search. We utilize the CLIP (Contrastive Language-Image Pretraining) from OpenAI, as the foundation for conducting these searches efficiently.
## What is CLIP (Contrastive Language-Image Pretraining) Model?
CLIP is a neural network that has undergone training using a wide array of (image, text) pairs. By leveraging natural language instructions, it excels in accurately predicting the most relevant text snippet corresponding to a given image. This proficiency stems from its training on a large dataset inclusive of image-caption pairs.

<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/CLIP.png">


CLIP achieves a unified embedding space (Latent Space) for images and text, facilitating direct comparisons between the two forms of data. This is achieved through training the model to minimize the distance between related images and texts while maximizing the separation between unrelated pairs. Thus, we can compare an image and a text by computing the cosine similarity between their respective embedding vectors.

<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/embeddings.png">


## Dataset
The dataset consists of 1807 images showcasing various objects:
**humans, cats, dogs, cars, bikes, horses and flowers**. You can find the analysis of our dataset in the `notebook/images_dataset_analysis.ipynb`. The images come in different file formats including JPG, PNG and BMP.

## Architecture

The following image describes the architecture of the designed solution. The whole solution runs on Docker containers. This decision was made to mitigate issues stemming from compatibility and dependencies in order to enhance the solution's reliability and consistency.
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/architecture.jpg">


The architecture is based on three different components:
### 1. FastAPI Container
   - **Role:** This container functions as the back-end server responsible for handling search queries. It processes textual queries and retrieves the relevant images from the dataset.
   - **Logic:** On the startup of the server, there's an automatic process that retrieves all images from the dataset. It then generates embeddings for each image and stores them individually in the Qdrant vector database. Note that this process may require some time to complete due to its complexity. 
  
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/startup.jpg">

After this process, the server becomes capable of accepting requests to search for images based on textual queries sent by the client app.
   - **Contents:**
      - `main.py`: the entry point to our back-end app.
      - `utils`:
        - `qdrant_vector_database.py`: contains a class designed for interacting with the Qdrant database. This class is responsible for managing tasks such as storing image embeddings in Qdrant and retrieving the most similar images based on textual queries. Note that here we use the built in **Cosine similarity** metric for Qdrant to make the process of retrieving faster.
        -  `model_conf.py`: serves the purpose of handling various aspects related to working with our model **(openai/clip-vit-base-patch32)**. It saves the model to the device and retrieves  its associated tokenizer and processor.
        -  `embeddings_handlers:`
            - `images_embeddings.py`: responsible for generating images embeddings.
            - `text_embeddings.py`: responsible for generating text embeddings.
### 2. ReactJs Container
   - **Role:** This container is the front-end of our solution. It is a straightforward and simple interface. Users can input text queries thanks to a search bar to find images similar to their query.
   - **Logic:** This component allows users to input a text query via the search bar. Once the user submits the query, it sends the text to the back-end server through the /search/images endpoint and awaits a response. Upon receiving the response, which consists of a list of URLs representing the most similar images to the query, it displays these images to the user.

### 3. Qdrant Container
   - **Role:** This container functions as a specialized database tailored for storing high-dimensional vectors, such as our image embeddings. Unlike traditional databases, it's optimized for efficiently querying these vectors. It employs various similarity search methods, including **Cosine Similarity**, which is the approach we used in this project for conducting searches based on similarity.

   - **Logic:** The container initially receives a list of embeddings for all the dataset images from the back-end, storing them within a collection. When the back-end requests a similarity search for a text query from the Qdrant Container, the Qdrant engine retrieves the image embeddings stored within the collection that have the highest cosine similarity with the provided text.

<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/search_similarity.jpg">

## Run Solution
In order to run the solution, follow these steps:
1. Build Docker images for our services. Please note that this process may take some time as it involves installing all the necessary dependencies:
    ```bash
     sudo docker-compose build
   ```
2.  Create and start the containers based on the corresponding images that were created and wait for all the containers to start: 
    ```bash
     sudo docker-compose up
    ```
3. Once the containers are all running, navigate to: `http://localhost:3000/` through your brower and start searching.

## Evaluation
### Accurate results
- **Query =** Black dog
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_1.png">

- **Query =** White shirt
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_2.png">

- **Query=** Green grass
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_3.png">

- **Query=** Red car
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_4.png">

### Inaccurate results
- **Query =** Two dogs
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate1.png">

- **Query =** A woman with blond hair
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate2.png">

- **Query=** White pants
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate3.png">

- **Query=** Two cats
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate4.png">
### Areas to improve
