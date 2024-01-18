import streamlit as st
from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a root `/` endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}



import torch
from omegaconf import OmegaConf
from ml_ops_detect_ai_generated_text.models.model import TextClassificationModel


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




def main():
    st.title("Detection of AI-generated Text")
    st.markdown("Under construction")

    st.markdown("Alternatively, you can upload a text file. Please make sure that the file is in .txt format.")
    tab1, tab2 = st.tabs(["Text input", "File upload"])
    valid_input = False
    with tab1:
        user_input = st.text_input("Please enter the text you want to check for AI-generated content")
        if user_input:
            valid_input = True

    with tab2:
        uploaded_file = st.file_uploader("Choose a file")
        # read the uploaded file
        if uploaded_file is not None:
            # Check if file is in .txt format
            if uploaded_file.type != "text/plain":
                st.error("Please upload a text file in .txt format")
                st.stop()
            uploaded_file.seek(0)
            user_input = uploaded_file.read().decode("utf-8")
            valid_input = True

    if not valid_input:
        st.stop()
    
    # Display text for user
    st.markdown("You entered the following text:")
    st.markdown(f"> {user_input}")

    # Display prediction
    st.markdown("The model predicts:")

    # Run the model
    prediction = predict(user_input)
    st.markdown(f"> {prediction}")

    st.markdown("To test a new input simply enter a new text or upload a new file.")



def entry_point():
    main()


if __name__ == "__main__":
    main() 


"""
streamlit run app/streamlitapp.py
"""