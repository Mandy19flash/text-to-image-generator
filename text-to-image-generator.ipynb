import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

# Load the Stable Diffusion model
# Note: This will download the model the first time you run it, which may take some time.
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda") # Move model to GPU if available

# Define a function to generate an image from a text prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt here:", placeholder="A white cat with blue eyes..."),
    outputs=gr.Image(label="Generated Image"),
    title="Image Generator"
)

# Launch the Gradio interface
iface.launch()
