import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2

# Function to load the TSV file and prepend the base directory
def load_tsv(tsv_file, base_dir):
    # Read the TSV file
    df = pd.read_csv(tsv_file, sep="\t", header=None, names=["image_path", "ground_truth"])
    
    # Prepend the base directory (up to `train` or `test`)
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(base_dir, x))
    
    return df

# Function to resize images and save them in the correct subdirectory
def resize_images_and_save(tsv_file, output_folder, base_dir):
    # Load the TSV file with image paths and ground truth
    df = load_tsv(tsv_file, base_dir)
    
    for _, row in df.iterrows():
        image_path = row["image_path"]
        print(f"Loading image from: {image_path}")  # Debugging print
        
        # Check if the image path ends with an extension (e.g., .png), add it if not
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path += ".png"  # Assuming all files are PNG if no extension is provided

        # Normalize the path to handle any OS-specific issues
        image_path = os.path.normpath(image_path)
        
        # Check if image exists, then load it using PIL (to handle PNG and other formats)
        try:
            image = Image.open(image_path).convert('L')  # Convert image to grayscale
        except Exception as e:
            print(f"Failed to load image: {image_path}. Error: {e}")  # If the image couldn't be loaded
            continue
        
        # Resize image to 64x64
        resized_image = image.resize((64, 64), Image.Resampling.LANCZOS)  # Resize to 64x64 using LANCZOS
        
        # Convert the image back to a numpy array for saving with OpenCV
        resized_image = np.array(resized_image)
        
        # Define the output folder (either `train` or `test`)
        subfolder = os.path.dirname(image_path).replace(base_dir, "").strip(os.sep)
        
        # Define the output path as the base `output_folder` followed by `train` or `test`
        output_subfolder = os.path.join(output_folder, subfolder)
        
        # Ensure the subfolder exists
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Save the resized image using OpenCV (to maintain compatibility with the output format)
        output_image_path = os.path.join(output_subfolder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, resized_image)
        print(f"Saved resized image to: {output_image_path}")

# Main function to preprocess the images for train and test datasets
def preprocess_data():
    # Define paths to your TSV files
    tsv_files = {
        "train": r"C:\Users\Dande\Desktop\Deep Learning Project\data\groundtruth_train.tsv",
        "test_2013": r"C:\Users\Dande\Desktop\Deep Learning Project\data\groundtruth_2013.tsv",
        "test_2014": r"C:\Users\Dande\Desktop\Deep Learning Project\data\groundtruth_2014.tsv",
        "test_2016": r"C:\Users\Dande\Desktop\Deep Learning Project\data\groundtruth_2016.tsv"
    }
    
    # Define output directory where resized images will be saved
    output_folder = r"C:\Users\Dande\Desktop\Deep Learning Project\output"
    
    # Ensure the 'train' and 'test' subdirectories exist in the output folder
    os.makedirs(os.path.join(output_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "test"), exist_ok=True)
    
    # Preprocess train data
    base_dir = r"C:\Users\Dande\Desktop\Deep Learning Project\data\train"
    resize_images_and_save(tsv_files["train"], os.path.join(output_folder, "train"), base_dir)
    
    # Preprocess test data for different years
    base_dir = r"C:\Users\Dande\Desktop\Deep Learning Project\data\test\2013"
    resize_images_and_save(tsv_files["test_2013"], os.path.join(output_folder, "test"), base_dir)
    
    base_dir = r"C:\Users\Dande\Desktop\Deep Learning Project\data\test\2014"
    resize_images_and_save(tsv_files["test_2014"], os.path.join(output_folder, "test"), base_dir)
    
    base_dir = r"C:\Users\Dande\Desktop\Deep Learning Project\data\test\2016"
    resize_images_and_save(tsv_files["test_2016"], os.path.join(output_folder, "test"), base_dir)

# Run the preprocessing function
if __name__ == "__main__":
    preprocess_data()
