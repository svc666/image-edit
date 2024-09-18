import PIL
import requests
import torch
import streamlit as st
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Set the model ID
model_id = "timbrooks/instruct-pix2pix"

# Load the model
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Function to download image from a URL
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Streamlit app layout
st.title("Image Transformation with Instruct Pix2Pix")

# Image upload functionality
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Text input for the transformation prompt
    prompt = st.text_input("Enter your prompt:", "turn him into cryboy")

    if st.button("Transform"):
        with st.spinner("Transforming..."):
            # Perform the image transformation
            transformed_image = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]
            st.image(transformed_image, caption="Transformed Image", use_column_width=True)

# Optional: Predefined example image download
if st.button("Use Example Image"):
    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    example_image = download_image(url)
    st.image(example_image, caption="Example Image", use_column_width=True)

    # Text input for the transformation prompt for the example image
    example_prompt = st.text_input("Enter your prompt for the example image:", "turn him into cryboy")

    if st.button("Transform Example"):
        with st.spinner("Transforming..."):
            transformed_example_image = pipe(example_prompt, image=example_image, num_inference_steps=10, image_guidance_scale=1).images[0]
            st.image(transformed_example_image, caption="Transformed Example Image", use_column_width=True)
