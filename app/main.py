from fastapi import FastAPI
import os
import uvicorn

# Create the FastAPI app
app = FastAPI()

# Define a root `/` endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}


import torch
from omegaconf import OmegaConf
from google.cloud import storage

import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class TextClassificationModel(pl.LightningModule):
    def __init__(self, model_name=None, num_labels=None, learning_rate=None) -> None:
        super(TextClassificationModel, self).__init__()
        self.model_name = model_name
        # Using a pretrained distilbert model, with a single linear layer on
        # top
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        # we only want to update the classifier, not the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        #
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name) #
        # NOTE: used?
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)



def download_from_bucket(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_model():
    # Path variables
    model_path = "models/2024-01-17/09-31-30/distilbert-base-uncased-epoch=00-val_loss=0.00.ckpt"
    config_path = "outputs/2024-01-17/09-31-30/.hydra/config.yaml"

    # Create directories for the paths
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Download from GCP bucket
    bucket = 'mlops_model_unique'
    download_from_bucket(bucket, 'deploy/model.ckpt', model_path)
    download_from_bucket(bucket, 'deploy/config.yaml', config_path)

    # Check if paths are present
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Model or config paths are not present")

    ### Load the config
    config = OmegaConf.load(config_path)
    ### Load the model
    state_dict = torch.load(model_path)
    # Create an instance of the model
    model = TextClassificationModel(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        learning_rate=config.training.learning_rate,
    )
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict'], strict=False)
    # strict=False because the model is saved with the lightning module, only match saved keys (i.e. classification layer)
    model.eval()
    return model


def format_prediction(class_idx):
    if class_idx == 0:
        return "A human being wrote this."
    elif class_idx == 1:
        return "A bunch of zeros and ones wrote this."

model = load_model()


@app.get("/predict")
def predict(text: str):
    # Tokenize the text
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True)
    # Make sure 'input_ids' is a tensor, not a list
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Run the model
    with torch.no_grad():
        prediction = model(input_ids, attention_mask=attention_mask)
    prediction = prediction.logits
    # Get the index of the maximum value in the prediction tensor
    predicted_class_index = torch.argmax(prediction).item()

    # Return the predicted class index as JSON
    #return {"predicted_class_index": predicted_class_index}
    return format_prediction(predicted_class_index)



"""
uvicorn app.main:app --reload
"""


def entry_point():
    #uvicorn.run(app, host="0.0.0.0", port=8080)
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)


if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8080)
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

