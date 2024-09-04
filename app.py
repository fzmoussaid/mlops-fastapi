from fastapi import FastAPI
import torch
from torchvision import transforms
from PIL import Image
import boto3
from pydantic import BaseModel
import ast
import pandas
import os

IMAGE_STORAGE_PATH = "vegetables/"
BUCKET_NAME = "classification-images-bucket"
PATH = "vegetables_classification_net.pth"
MAPPING_FILENAME = "mapping.csv"

class Payload(BaseModel):
    input: str

def load_model(path):
    model = torch.jit.load(path)
    return model

app = FastAPI()
model = load_model(PATH)
s3_client = boto3.client("s3", aws_access_key_id=os.environ["AWS_ACCESS_KEY"], aws_secret_access_key=os.environ["AWS_SECRET_KEY"] )

def preprocess_data(input):
    image_transform = transforms.Compose(
        [transforms.Resize((50,50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.4301, std=0.2203)]
    )
    preprocessed_input = image_transform(input).unsqueeze(0)
    return preprocessed_input

def get_prediction(input):
    model.eval()
    data = preprocess_data(input)
    with torch.no_grad():
        output = model(data)
        _, prediction = torch.max(output, 1)
    return prediction

def load_mapping():
    s3_client.download_file(BUCKET_NAME, MAPPING_FILENAME, MAPPING_FILENAME)
    df = pandas.read_csv(MAPPING_FILENAME)
    classes = df.to_dict()
    return classes['class_name']

@app.get("/")
def read_root():
    return {"Content": "Vegetables Classification"}

@app.get("/ping")
def ping():
    return "pong"

@app.post("/invocations")
def invoke(payload: Payload):
    payload = payload.json()
    payload = ast.literal_eval(payload)
    image_name = payload["input"] + ".jpg"
    c_image = "c_image.jpg"

    s3_client.download_file(BUCKET_NAME, IMAGE_STORAGE_PATH + image_name, c_image)
    image = Image.open(c_image)
    prediction = get_prediction(image)
    classes_mapping = load_mapping()
    predicted_class = classes_mapping[prediction.item()]
    return { "predicted_class": predicted_class }