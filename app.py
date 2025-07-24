import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

st.set_page_config(page_title="BaldSight", layout="wide")

# Environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load Gemini model 
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

#  PyTorch  
@st.cache_resource
def load_trained_model():
    model = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # You trained on 5 classes
    model.load_state_dict(torch.load("baldsight_resnet18.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_trained_model()

#  Prediction 
def predict_stage(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        confidence_value = confidence.item() * 100
        stage_message = f"Predicted Baldness Stage: {predicted_class.item()}"
        confidence_message = f"Confidence: {confidence_value:.2f}%"
        if confidence_value < 90:
            confidence_message += " (low confidence)"
        return stage_message, confidence_message

# Mapping
def get_stage_from_text(text):
    prompt = f"""Based on the following user description of their hair loss, identify the Norwood scale baldness stage (0 to 4). Just return the stage number.
Description: {text}"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

#  Sidebar 
with st.sidebar:
    st.markdown("### About BaldSight")
    st.markdown(
        """
        This tool uses a fine-tuned ResNet18 model to detect the Norwood scale stage of baldness from a scalp image or textual description.

        **Instructions:**
        - Upload a clear image (top or front view).
        - Or enter a detailed description.
        - Predictions under 90% confidence are marked as such.
        """
    )

# UI
st.markdown("<h1 style='text-align: center;'>BALDSIGHT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-Powered Baldness Stage Detection</h4>", unsafe_allow_html=True)

# Image Upload
st.subheader("Upload a scalp image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    stage, confidence = predict_stage(image)
    st.success(stage)
    st.info(confidence)

# Text Input 
st.subheader("Or describe your hair loss")
text_input = st.text_input("Enter a description of your hair loss:")

if st.button("Get Baldness Stage from Description"):
    if text_input:
        stage = get_stage_from_text(text_input)
        st.success(f"Predicted Stage (Text-Based): {stage}")
    else:
        st.warning("Please enter a description.")

