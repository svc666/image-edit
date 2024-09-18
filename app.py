
import streamlit as st
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Load the model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("cpu")  # Load the model to CPU
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Streamlit app
st.title("Image Transformation with Instruct Pix2Pix")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Get prompt from user input
prompt = st.text_input("Enter your prompt:")

if uploaded_file and prompt:
    # Load the image from the uploaded file
    image = PIL.Image.open(uploaded_file)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Generate images based on the prompt
    with st.spinner("Generating image..."):
        images = pipe(prompt, image=image, num_inference_steps=5, image_guidance_scale=1).images

    # Display the generated image
    st.image(images[0], caption="Transformed Image", use_column_width=True)
