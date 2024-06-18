import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import io

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

st.title("Image Captioning")

# Function to generate caption
def generate_caption(image, conditional_text=None):
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Sidebar for uploading or capturing image
option = st.sidebar.selectbox("Select Input Method", ["Upload a Photo", "Capture a Photo"])

if option == "Upload a Photo":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Generating caption...")
        caption = generate_caption(image)
        st.write(f"Caption: {caption}")

elif option == "Capture a Photo":
    st.write("Please use a device with a camera to capture a photo.")
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert('RGB')
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.write("Generating caption...")
        caption = generate_caption(image)
        st.write(f"Caption: {caption}")

# Optional: Add a text input for conditional captioning
st.write("Optional: Add some text to condition the caption generation.")
text_input = st.text_input("Enter some text:")
if st.button("Generate Caption"):
    if image:
        conditional_caption = generate_caption(image, text_input)
        st.write(f"Conditional Caption: {conditional_caption}")
    else:
        st.write("Please upload or capture an image first.")
