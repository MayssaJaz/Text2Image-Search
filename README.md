# Text2Image-Search

## Purpose of the Project
This project aims create a search engine designed specifically for retrieving images from a dataset using textual queries, a process commonly known as text-to-image search. We utilize the CLIP (Contrastive Language-Image Pretraining) from OpenAI, as the foundation for conducting these searches efficiently.
## What is CLIP (Contrastive Language-Image Pretraining) Model?
CLIP is a neural network that has undergone training using a wide array of (image, text) pairs. By leveraging natural language instructions, it excels in accurately predicting the most relevant text snippet corresponding to a given image. This proficiency stems from its training on a large dataset inclusive of image-caption pairs.

<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/CLIP.png">


CLIP achieves a unified embedding space (Latent Space) for images and text, facilitating direct comparisons between the two forms of data. This is achieved through training the model to minimize the distance between related images and texts while maximizing the separation between unrelated pairs. Thus, we can compare an image and a text by computing the cosine similarity between their respective embedding vectors.

<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/embeddings.png">


## Dataset
The dataset consists of 1803 images showcasing various objects:
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
1. If you want to run the original model, remove the following line from `docker-compose.yml`:
   ```bash
    - MODEL_PARAMS_PATH=weights/optimized_model.pt
   ```
   Otherwise, download the weights of the fine-tuned model (link inside the resources section below) and place them inside `back/weights`.
   
3. Build Docker images for our services. Please note that this process may take some time as it involves installing all the necessary dependencies:
    ```bash
     sudo docker-compose build
   ```
4.  Create and start the containers based on the corresponding images that were created and wait for all the containers to start: 
    ```bash
     sudo docker-compose up
    ```
5. Once the containers are all running, navigate to: `http://localhost:3000/` through your brower and start searching.

## Evaluation
### Before Fine-tuning
#### Accurate results
- **Query =** Black dog
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_1.png">


- **Query =** White shirt
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_2.png">


- **Query=** Green grass
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_3.png">


- **Query=** Red car
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/accurate/accurate_4.png">

#### Inaccurate results
- **Query =** Two dogs
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate1.png">
It's evident that in the third image, there is both a dog and a cat present. This suggests that the cat was mistakenly perceived as a dog in our case.


- **Query =** A woman with blond hair
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate2.png">
A more complex query generated irrelevant results. As we can see, the only image that is accurate is the second one.


- **Query=** White pants
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate3.png">
In this scenario, it's evident that the color of the shirt and the pants were confused, resulting in inaccurate outcomes in the 3rd, 4th, and 5th images.


- **Query=** Two cats
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/inaccurate/inaccurate4.png">
While the outcomes of this query was not as poor as those from previous queries, it's important to emphasize that the 5th image is entirely inaccurate as it doesn't even represent a picture of a cat.

##### Important remark
Just to clarify, the inaccurate queries were not due to insufficient data. Each query had a minimum of five matching images. Therefore, we decided to analyze a dataset containing 1803 instances to verify if there were indeed images corresponding to the query before evaluating it with our search engine. 

##### Explanation 
For the innacurate query results containing a number, the OpenAI CLIP model may encounter challenges with systematic tasks, particularly counting the number of objects (See: [OpenAI CLIP](https://openai.com/research/clip) ). In addition, the model may face difficulties in understanding the context of queries such as confusing the colors of the shirt and the pants, especially when dealing with data it hasn't been exposed to during training.

### After Fine-tuning: 
#### Steps
After generating our labeled dataset inside `docs/labeled_data/data.csv` (See the process: `docs/notebook/captions_generation.ipynb`) that has two columns the file name and the caption of each image in our dataset using the OpenClip **CoCa (Contrastive Captioners)** that was designed for generating images captions.), we managed to fine-tune our model using the generated captions to better enhance text-to-image similarity. This refinement process is detailed in `docs/notebook/fine_tune_clip_model.ipynb`.Note that we used Contrastive Loss, which serves the purpose of minimizing the cosine similarity between embeddings of similar texts and images while simultaneously maximizing the dissimilarity between embeddings of different texts and images. This approach is aimed at preventing the model from forgetting the patterns it learned during its pre-training stage (See: [Finetune like you pretrain](https://arxiv.org/pdf/2212.00638.pdf) ).

#### Results
We conduct a repetition of the previous queries and proceed to evaluate their performance through testing:
- **Query =** Black dog
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/6.png">

- **Query =** White shirt
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/5.png">

- **Query=** Green grass
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/7.png">

- **Query=** Red car
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/8.png">

- **Query =** A woman with blond hair
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/1.png">

- **Query =** Two dogs
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/2.png">

- **Query=** White pants
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/4.png">

- **Query=** Two cats
<img src="https://github.com/MayssaJaz/Text2Image-Search/blob/main/docs/results/fine_tuned/3.png">

#### Discussion
After testing the same queries with the new version of the model following fine-tuning, significant improvements are evident across most queries. The model now excels in accurately identifying two dogs and two cats. Additionally, it successfully recognizes white pants and a woman with blond hair. However, there is a slight inconsistency in identifying the pattern of white shirts. In the first and second images, the model identified two blue shirts with a hint of white, which is not entirely incorrect. To enhance the model's proficiency in recognizing such patterns, incorporating additional data containing these features (such as white shirts) and further fine-tuning them will undoubtedly improve its effectiveness.



### Areas to improve

- Ensure the generation of queries that are human-generated and cover a broad spectrum of topics and concepts that are relevant to the dataset. These queries should encompass diverse aspects, including concepts such as colors, shapes, and proximity of two objects as well as more intricate queries requiring contextual understanding, such as those that are prone to misinterpretation in evaluations (e.g the color of a shirt = the color of pants in our case).

- For the evaluation phase, since we are dealing with a search engine problem, we can use metrics that are widely used to measure how relevant are documents to the a specific query such as **Precision@K**, **Recall@K** and **Normalized Discounted Cumulative Gain@K (nDCG@K)**.
## Next Step

- [x]  Fine-tune our model by using the labeled dataset generated in  `docs/labeled_data/data.csv`

- [ ] Assess its performance across various queries we're preparing.

- [ ] Expand our dataset by acquiring more data and labeling it, followed by retraining the model as part of an MLOps pipeline.

- [ ] Exlore alternative models such as the **BLIP** model and benchmark the results to assess their effectiveness based on various criterias (Relevance / Speed / Resource Consumption...).
## Resources
- **Kaggle dataset:** https://www.kaggle.com/datasets/pavansanagapati/images-dataset
- **CLIP model on hugging face:** https://huggingface.co/openai/clip-vit-base-patch32
- **CLIP model paper:** https://arxiv.org/pdf/2103.00020.pdf
- **Qdrant similarity search:** https://qdrant.tech/documentation/concepts/search/
- **Qdrant github repositories:** https://github.com/qdrant/qdrant_demo +  https://github.com/qdrant/qdrant-client
- **Image classification Benchmark (for generating captions)** https://paperswithcode.com/sota/image-classification-on-imagenet
- **OpenClip repository:** https://github.com/mlfoundations/open_clip/tree/main
- **BLIP model:** https://huggingface.co/docs/transformers/model_doc/blip
- **Finetune like you pretrain:** https://arxiv.org/pdf/2212.00638.pdf
- **Link to parameters:** https://drive.google.com/file/d/1Ym0BHz33dY_TFHuUCADM8186e7lFpoad/view?usp=drive_link
