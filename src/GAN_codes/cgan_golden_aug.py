# =================================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =================================================================================
import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# --- Configuration ---
BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET"
# Define which classes you want to process.
CLASSES_TO_PROCESS = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]

# Define the names of your input and output folders.
INPUT_FOLDER_NAME = "golden_set"  # The folder with your ~100 high-quality images.
OUTPUT_FOLDER_NAME = "augmented_for_gan" # The folder where the new images will be saved.

# Define how many new images to create from each original image.
# If you have 100 original images, setting this to 10 will create 1000 total images.
IMAGES_PER_ORIGINAL = 10 
IMG_SIZE = (128, 128) # The size your GAN expects.

# =================================================================================
# SECTION 2: AUGMENTATION FUNCTIONS
# =================================================================================

@tf.function
def augment_image(image):
    """Applies a series of random geometric augmentations to a single image."""
    # Convert image to float32 for augmentation functions
    image = tf.cast(image, tf.float32)
    
    # 50% chance of flipping horizontally
    image = tf.image.random_flip_left_right(image)
    
    # Randomly rotate by -20 to +20 degrees.
    # tf.image.rot90 is too restrictive, so we use a more flexible method.
    # Note: tf.keras.layers.RandomRotation is easier but this works inside @tf.function
    # For simplicity, we will use tf.image.rot90 which rotates by multiples of 90 degrees.
    # For finer rotations, a library like `imgaug` or a custom function is needed.
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Randomly adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Randomly shift the image slightly.
    # This is a more complex operation, for simplicity we will skip it in this version,
    # as flips and rotations provide significant variety.

    # Clip values to be in the valid [0, 255] range for saving.
    image = tf.clip_by_value(image, 0.0, 255.0)
    
    return tf.cast(image, tf.uint8)

# =================================================================================
# SECTION 3: MAIN PROCESSING LOOP
# =================================================================================

print("▶️ Starting pre-augmentation process...")

# Loop over each class you want to augment.
for class_name in CLASSES_TO_PROCESS:
    print(f"\n--- Processing class: {class_name} ---")
    
    # Define input and output paths
    input_dir = os.path.join(BASE_PATH, class_name, INPUT_FOLDER_NAME)
    output_dir = os.path.join(BASE_PATH, class_name, OUTPUT_FOLDER_NAME)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    # Find all images in the input directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(input_dir, "*.JPG"))

    if not image_files:
        print(f"⚠️ Warning: No images found in {input_dir}. Skipping.")
        continue
        
    print(f"Found {len(image_files)} original images. Generating {IMAGES_PER_ORIGINAL} augmentations for each.")
    
    total_generated = 0
    # Loop over each original image
    for i, file_path in enumerate(image_files):
        try:
            # Read and decode the image
            img_tensor = tf.io.read_file(file_path)
            img_tensor = tf.image.decode_image(img_tensor, channels=3, expand_animations=False)
            img_tensor = tf.image.resize(img_tensor, IMG_SIZE)
            
            # Generate N augmented versions of this single image
            for j in range(IMAGES_PER_ORIGINAL):
                augmented_tensor = augment_image(img_tensor)
                
                # Convert tensor to a PIL Image for saving
                augmented_image = Image.fromarray(augmented_tensor.numpy())
                
                # Create a unique filename
                original_filename = os.path.basename(file_path)
                name, ext = os.path.splitext(original_filename)
                new_filename = f"{name}_aug_{j}{ext}"
                
                # Save the new image
                augmented_image.save(os.path.join(output_dir, new_filename))
                total_generated += 1

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")

    print(f"✅ Finished processing {class_name}. Generated {total_generated} new images.")

print("\n🎉 Pre-augmentation complete for all classes!")
