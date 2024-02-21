FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y unzip && pip install --no-cache-dir -r requirements.txt && pip install torch --index-url https://download.pytorch.org/whl/cpu && gdown 1ePX5hiG55M0cDiJv8oM0Vh28ADCszgQA && unzip images_project.zip

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]