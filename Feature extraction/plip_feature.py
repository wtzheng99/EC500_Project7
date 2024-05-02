import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the model and processor
model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

# Directory containing the images
image_dir = 'annotated_patches/image'
save_dir = 'annotated_patches/plip_features'

# Iterate over each file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):  # Checks for PNG images, adjust if different formats are used
        # Load and process the image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)

        # Forward pass through the model to get the image features
        outputs = model.get_image_features(**inputs)

        # Save the output tensor to a file
        torch.save(outputs, os.path.join(save_dir, f"{filename[:-4]}.pt"))  # Removes .png and adds .pt

        print(f"Saved features for {filename} as {filename[:-4]}.pt")
