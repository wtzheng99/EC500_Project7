import cv2
import json
import os
import torch
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np

folder_dir = 'annotated_patches'
image_dir = 'annotated_patches/image'
json_dir = 'annotated_patches/json'
feature_dir = 'annotated_patches/resnet_features'


# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True, progress=False)
model.eval()

# Modify the model to remove the last fully connected layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))


def crop_images(large_image_path, bbox_list):
    cropped_img = []
    large_image = cv2.imread(large_image_path)
    if large_image is None:
        print("Error loading image")
        return []
    
    large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    for bbox in bbox_list:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        cropped_img.append(large_image[x1:x2, y1:y2, :])
        
    return cropped_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


for json_filename in os.listdir(json_dir):
    if json_filename.endswith('.json'):
        # Construct the full path to the JSON file
        json_path = os.path.join(json_dir, json_filename)
        
        # Derive the corresponding image file path
        image_basename = os.path.splitext(json_filename)[0] + '.png'  # Replace with '.jpg' if your images are in jpg format
        image_path = os.path.join(image_dir, image_basename)

        # Check if the corresponding image file exists
        if not os.path.exists(image_path):
            print(f"No matching image for {json_filename}")
            continue

        # Read the JSON file and extract bounding boxes
        bbox_list = []
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            mag_info = data['mag']
            nuc_info = data['nuc']
            for inst in nuc_info:
                inst_info = nuc_info[inst]
                inst_bbox = inst_info['bbox']
                bbox_list.append(inst_bbox)

        # Now process the matching image file
        cropped_img = crop_images(image_path, bbox_list)

        # Create a placeholder for the features
        features = torch.empty((len(cropped_img), 2048), device=device)  # Corrected dimension

        # Define normalization mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Process the images
        for i, image in enumerate(cropped_img):
            image = cv2.resize(image, (224, 224))
            tensor = TF.to_tensor(image).to(device)
            tensor = TF.normalize(tensor, mean=mean, std=std).unsqueeze(0)  # Normalize and add batch dimension

            # Forward pass to get features
            with torch.no_grad():
                output = model(tensor)
                features[i, :] = output.squeeze()

        # Verify the size of the feature tensor
        # print(features.size())
        feature_basename = os.path.splitext(json_filename)[0] + '.pt'
        feature_saved_path = os.path.join(feature_dir, feature_basename)
        torch.save(features.cpu(), feature_saved_path)
