# =================================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =================================================================================
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import time

print(f"TensorFlow Version: {tf.__version__}")

# Configure GPU memory growth to prevent allocation errors.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Enabled memory growth for {len(gpus)} GPU(s).")
  except RuntimeError as e:
    print(f"❌ Error setting memory growth: {e}")

# --- Configuration ---
# The base path where your class folders are located.
BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET"
# The folder where the newly generated images will be saved.
OUTPUT_FOLDER_NAME = "augmented_for_gan_new"

# The size of the random noise vector (must match what the models were trained on).
LATENT_DIM = 128

# =================================================================================
# SECTION 2: DEFINE MODELS AND IMAGE COUNTS
# =================================================================================
# path = /mnt/shared/College_work_data/College Work/Projects/CNN_confrence_paper/GAN_models/gan_wgangp_blight_output/checkpoints/blight_generator_epoch_0300.h5
# ✅ STEP 1: UPDATE THESE PATHS
# You MUST replace these placeholder paths with the actual paths to your best
# generator checkpoint files (.h5) for each class.
MODELS_TO_USE = {
    "BLAST": "/mnt/shared/College_work_data/College Work/Projects/CNN_confrence_paper/GAN_models/gan_wgangp_blast_output/checkpoints/blast_generator_epoch_0300.h5",
    "BLIGHT": "/mnt/shared/College_work_data/College Work/Projects/CNN_confrence_paper/GAN_models/gan_wgangp_blight_output/checkpoints/blight_generator_epoch_0300.h5",
    "BROWNSPOT": "/mnt/shared/College_work_data/College Work/Projects/CNN_confrence_paper/GAN_models/gan_wgangp_brownspot_output/checkpoints/brownspot_generator_epoch_0300.h5"
}

# --- Define how many images to generate for each class ---
# Based on your dataset numbers:
# Healthy: 8116 (Target)
# Blast: 6831
# Blight: 3610
# Brownspot: 5454
IMAGES_TO_GENERATE = {
    "BLAST": 8116 - 6831,      # Generate 1285 images
    "BLIGHT": 8116 - 3610,     # Generate 4506 images
    "BROWNSPOT": 8116 - 5454   # Generate 2662 images
}

# =================================================================================
# SECTION 3: IMAGE GENERATION
# =================================================================================

def generate_and_save_images(class_name, model_path, num_images):
    """Loads a generator model and saves a specified number of generated images."""
    
    print(f"\n--- Processing class: {class_name} ---")
    
    # --- Load the saved generator model ---
    print(f"▶️ Loading generator model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}. Skipping this class.")
        return
        
    try:
        generator = tf.keras.models.load_model(model_path)
        print("✅ Generator loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Prepare Output Directory ---
    save_dir = os.path.join(BASE_PATH, class_name, OUTPUT_FOLDER_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"▶️ Generating {num_images} new images...")
    start_time = time.time()
    
    # --- Generate and Save Images in Batches ---
    batch_size = 64 # Generate in batches to conserve memory
    images_saved = 0
    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)
        
        # Create random noise to feed the generator.
        noise = tf.random.normal([current_batch_size, LATENT_DIM])
        
        # Generate images.
        generated_images = generator(noise, training=False)
        
        # Rescale pixel values from [-1, 1] back to [0, 255].
        generated_images = tf.cast(generated_images, tf.float32)
        generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)
        
        # Save each image in the batch.
        for j, img_array in enumerate(generated_images):
            img = Image.fromarray(img_array)
            img_index = i + j
            # Save as PNG for lossless quality.
            img.save(os.path.join(save_dir, f'gan_generated_{class_name.lower()}_{img_index:04d}.png'))
            images_saved += 1
            
    end_time = time.time()
    print(f"✅ Successfully saved {images_saved} new images to {save_dir} in {end_time - start_time:.2f} seconds.")


# --- Run the generation process for each class defined above ---
for class_name, num_to_generate in IMAGES_TO_GENERATE.items():
    model_path = MODELS_TO_USE.get(class_name)
    if model_path:
        generate_and_save_images(class_name, model_path, num_to_generate)
    else:
        print(f"⚠️ Warning: No model path defined for class '{class_name}'. Skipping.")

print("\n🎉 All synthetic images have been generated and saved!")
print("Your dataset is now balanced. You can train your classification model.")
