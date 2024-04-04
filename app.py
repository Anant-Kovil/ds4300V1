import streamlit as st
from io import StringIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone, PodSpec
from PIL import Image
import torch
from dotenv import load_dotenv
import os

load_dotenv()

# setup file uploader
file = st.file_uploader

# setup processor and model to generate embeddings for image
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

connect = Pinecone(api_key = os.getenv('api_key'))
index = connect.Index('image-embeddings')

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs["input_ids"] = torch.ones(1, 1, dtype=torch.long)
    outputs = model(**inputs)
    image_embedding = outputs.image_embeds.cpu().detach().numpy()

    results = index.query(
        vector=image_embedding.tolist(),
        top_k=10,
        include_metadata=True
    )
    names = [list(match['metadata'].values())[0] for match in results['matches']]

    st.write(names)
