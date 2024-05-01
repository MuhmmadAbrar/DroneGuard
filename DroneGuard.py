import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet
import pandas as pd
import os

# Function to load the trained model
@st.cache_resource()
def load_segmentation_model(model_path):
    map_location = torch.device('cpu')  # Load the model on CPU
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint)
    return model

# Function to preprocess the input image for segmentation
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

def segment_image(model, image):
    # Preprocess the input image
    input_image = preprocess_image(image)
    
    # Perform inference
    with torch.no_grad():
        model.eval()
        output_mask = model(input_image)
        predicted_mask = (output_mask.sigmoid() > 0.5).float()
    
    # Convert the output mask to numpy array
    predicted_mask = predicted_mask.squeeze().cpu().numpy()
    
    # Resize the predicted mask to match the original image size
    original_width, original_height = image.size
    predicted_mask = cv2.resize(predicted_mask, (original_width, original_height))
    
    # Ensure the datatype of the mask is uint8
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    
    return predicted_mask

def visualize_segmentation(original_image, segmented_mask):
    original_image = np.array(original_image)
    
    # Create a copy of the original image for the segmented image
    segmented_image = np.copy(original_image)
    
    # Create a light red mask for water bodies
    water_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color
    
    # Apply the water color to the segmented regions
    segmented_image[segmented_mask > 0] = np.dstack((segmented_mask[segmented_mask > 0] * water_color[0],
                                                      segmented_mask[segmented_mask > 0] * water_color[1],
                                                      segmented_mask[segmented_mask > 0] * water_color[2]))
    
    # Plot original and segmented image
    st.image(segmented_image, use_column_width=True, caption="Segmented Map")
    
    # Get coordinates of water bodies
    water_coordinates = np.argwhere(segmented_mask > 0)
    
    # Create a Pandas DataFrame for coordinates
    water_df = pd.DataFrame(water_coordinates, columns=['X', 'Y'])
    
    # Calculate percentage of water content
    total_pixels = segmented_mask.size
    water_pixels = np.count_nonzero(segmented_mask)
    water_percentage = (water_pixels / total_pixels) * 100
    
    # Display the first 20 water coordinates
    st.write("First 20 Water Coordinates (X, Y):")
    st.table(water_df.head(20).style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

def stitch_images(images):
    # Create a stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch the images
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        # Convert the result to RGB format
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    else:
        return None

def main():
    st.title("Drone Map Generator and Image Segmentation")

    # Upload images for stitching
    uploaded_files = st.file_uploader("Upload drone images for map stitching", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Display uploaded images
        st.subheader("Uploaded Images for Map Stitching:")
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        # Stitch images on button click
        if st.button("Generate Map"):
            images = [cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR) for uploaded_file in uploaded_files]
            # Stitch images
            result = stitch_images(images)
            if result is not None:
                # Display the stitched map
                st.subheader("Stitched Map:")
                st.image(result, use_column_width=True, caption="Stitched Map")
                
                # Convert stitched map to PIL Image
                stitched_map_pil = Image.fromarray(result)
                # Perform segmentation on the stitched map
                segmentation_model_path = "./best_model.pth"
                segmentation_model = load_segmentation_model(segmentation_model_path)
                segmented_mask = segment_image(segmentation_model, stitched_map_pil)
                # Visualize the segmented image
                visualize_segmentation(stitched_map_pil, segmented_mask)
            else:
                st.warning("Stitching failed due to lack of overlap between images.")
                # Perform segmentation on the first image
                first_image = Image.open(uploaded_files[0])
                segmentation_model_path = "./best_model.pth"
                segmentation_model = load_segmentation_model(segmentation_model_path)
                segmented_mask = segment_image(segmentation_model, first_image)
                st.subheader("Segmentation of the First Image:")
                visualize_segmentation(first_image, segmented_mask)

if __name__ == "__main__":
    main()