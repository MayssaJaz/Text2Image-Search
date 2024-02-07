# Text2Image-Search
## Purpose of the Project
This project aims create a search engine designed specifically for retrieving images from a dataset using textual queries, a process commonly known as text-to-image search. We utilize the CLIP (Contrastive Language-Image Pretraining) from OpenAI, as the foundation for conducting these searches efficiently.
## What is CLIP (Contrastive Language-Image Pretraining) Model?
CLIP is a neural network that has undergone training using a wide array of (image, text) pairs. By leveraging natural language instructions, it excels in accurately predicting the most relevant text snippet corresponding to a given image. This proficiency stems from its training on a large dataset inclusive of image-caption pairs.

CLIP achieves a unified embedding space for images and text, facilitating direct comparisons between the two forms of data. This is achieved through training the model to minimize the distance between related images and texts while maximizing the separation between unrelated pairs.



## Architecture

The following 
### 1. FastAPI Backend Microservice
The core of the project is a FastAPI backend microservice responsible for handling incoming requests, processing fingerprint data, and executing a matching algorithm.

### 2. MySQL Server
The microservice is connected to a MySQL server that hosts a database containing fingerprints. This database stores the target fingerprint images against which incoming fingerprints are matched.
## Steps
### Image Pre-Processing Workflow
The Fingerprint Matching Microservice processes input fingerprint images through a series of steps to enhance the quality of patterns, improve matching accuracy, and ensure effective comparison against target fingerprints. The following steps outline the image pre-processing workflow:

- 1. Background Removal to ensure that we focus only on fingerprints details.

- 2. Adaptive Sharpening Filter to  enhance the contrast and sharpen the details within the image.


- 3. 4-Step Enhancement Process

  - Normalize the Image
  - Find Region of Interest (ROI)
  - Estimate Local Orientation of Ridges
  - Compute Major Frequency of Ridges
  - Enhance fingerprint image via oriented filters

### Minutiae Feature Extraction
This step may take a while depending on the image. It generates a **result.png** file that contains the input image with minutiae points: red circles for ridge endings and blue circles for bifurcations.
### Matching Fingerprint
Upon successful features extraction from input image, we compare the **result.png** image with all the images contained inside the **target_folder**. The comparison is done using SIFT (Scale-Invariant Feature Transform) to extract keypoints and their descriptors from the image then we use FLANN (Fast Library for Approximate Nearest Neighbors) to compute the score for each target image.
## How to run solution?
### With Docker
- Run the following command inside the root folder:
   ```bash
     sudo docker-compose build
   ```
This command is going to create two containers a MySQL container that contains the database for target fingerprints and a FastAPI container linked to the first container. This command also installs all the necessary requirements for this projects which are contained inside the **requirements.txt** file.
- Run the following command inside the same folder and wait for the FastAPI server to start running:
   ```bash
     sudo docker-compose up 
   ```
- Open your web browser and go to http://localhost:8000/docs to access the FastAPI Swagger
- Execute the first request one time only to store all fingerprints inside the database.
- Go to the matching request, select your input fingerprint image to find its match and execute your request (this process can take a while depending on the image since it contains the preprocessing, the feature extraction and matching.)
### On Local Machine
- Start with configuring your environment variable
  ```env
     export DATABASE_URL=mysql+mysqlconnector://your_username:your_password@your_host:your_port/fingerprint
     ```
- Run the following command inside the **code** folder to install the different requirements:
  ```bash
     pip install -r requirements.txt
   ```
- Create a new database with MySQL server called: **fingerprint**.
- Launch the app using:
  ```bash
     uvicorn main:app --reload
   ```
- Open your web browser and go to http://localhost:8000/docs to access the FastAPI Swagger
- Execute the first request one time only to store all fingerprints inside the database.
- Go to the matching request, select your input fingerprint image to find its match and execute your request (this process can take a while depending on the image since it contains the preprocessing, the feature extraction and matching.)

## Remark

The **Swagger** folder contains images that demonstrate how to use it and the output of the matching process.