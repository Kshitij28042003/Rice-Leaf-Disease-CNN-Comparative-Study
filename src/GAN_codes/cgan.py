# =================================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =================================================================================
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import time
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")

# --- Configuration ---
# Define the path to your dataset and the class names.
BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
NUM_CLASSES = len(CLASSES)

# --- GAN & Image Parameters ---
# GANs are computationally expensive. 128x128 is a good starting point.
IMG_SIZE = 128
CHANNELS = 3
# The latent dimension is the size of the random noise vector fed to the generator.
LATENT_DIM = 128
BATCH_SIZE = 32

# --- Training Parameters ---
# GANs need many epochs to learn. Start with a moderate number and increase if needed.
EPOCHS = 100
# These are standard Adam optimizer parameters for GANs.
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5

# --- Output Paths ---
# This script will create a directory to save sample images and the final model.
OUTPUT_DIR = "gan_training_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
GENERATOR_SAVE_PATH = os.path.join(OUTPUT_DIR, "cgan_generator.h5")

# =================================================================================
# SECTION 2: DATA LOADING & PREPARATION
# =================================================================================
print("▶️ Loading and preprocessing dataset...")

def load_dataset():
    """Loads all image filepaths and labels from the 'augmented' subdirectories."""
    all_filepaths, all_labels = [], []
    for idx, class_name in enumerate(CLASSES):
        # ✅ MODIFIED: Path now points directly to the 'augmented' subfolder.
        class_folder = os.path.join(BASE_PATH, class_name, "augmented")
        # Find all common image file types.
        files = glob.glob(os.path.join(class_folder, "*.jpg")) + \
                glob.glob(os.path.join(class_folder, "*.jpeg")) + \
                glob.glob(os.path.join(class_folder, "*.png"))
        all_filepaths.extend(files)
        all_labels.extend([idx] * len(files))
    
    def process_img(filepath):
        """Reads and preprocesses a single image."""
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # Normalize images to the range [-1, 1]. This is crucial for GANs,
        # especially when using 'tanh' activation in the generator's final layer.
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    # Create a high-performance tf.data pipeline.
    filepaths_ds = tf.data.Dataset.from_tensor_slices(all_filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int32))
    
    images_ds = filepaths_ds.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Zip the images and labels together, shuffle, batch, and prefetch.
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))
    dataset = dataset.shuffle(buffer_size=len(all_filepaths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Dataset loaded with {len(all_filepaths)} images from 'augmented' folders.")
    return dataset

train_dataset = load_dataset()

# =================================================================================
# SECTION 3: CONDITIONAL GAN (CGAN) MODEL ARCHITECTURE
# =================================================================================

# --- The Generator ---
# It takes random noise and a class label, and outputs a fake image.
def build_generator():
    """Builds the Generator model."""
    noise_input = Input(shape=(LATENT_DIM,))
    label_input = Input(shape=(1,), dtype='int32')
    
    # The Embedding layer is the key to making the GAN conditional.
    # It turns the integer label (e.g., 2) into a dense vector that can be processed.
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(4 * 4)(label_embedding)
    label_embedding = layers.Reshape((4, 4, 1))(label_embedding)

    # Process the noise vector into a small feature map.
    noise_path = layers.Dense(4 * 4 * 256, use_bias=False)(noise_input)
    noise_path = layers.BatchNormalization()(noise_path)
    noise_path = layers.LeakyReLU()(noise_path)
    noise_path = layers.Reshape((4, 4, 256))(noise_path)

    # Concatenate the processed noise and the class label information.
    merged = layers.Concatenate()([noise_path, label_embedding])

    # A series of Conv2DTranspose (or "deconvolution") layers to upsample the feature map
    # into a full-sized image.
    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged) # 8x8
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x) # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(4, 4), padding='same', use_bias=False)(x) # 64x64
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    # The final layer produces the image. 'tanh' activation scales pixels to [-1, 1].
    output_image = layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x) # 128x128

    return Model([noise_input, label_input], output_image, name="Generator")

# --- The Discriminator ---
# It takes an image and a label, and predicts if the image is real or fake for that class.
def build_discriminator():
    """Builds the Discriminator model."""
    label_input = Input(shape=(1,), dtype='int32')
    # Embed the label and reshape it to the size of the image so they can be concatenated.
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(IMG_SIZE * IMG_SIZE)(label_embedding)
    label_embedding = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_embedding)
    
    image_input = Input(shape=[IMG_SIZE, IMG_SIZE, CHANNELS])
    
    # Concatenate the image and the label feature map.
    merged = layers.Concatenate()([image_input, label_embedding])

    # A series of Conv2D layers to analyze the image and extract features.
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    
    # The final output is a single logit (a raw score). It is not passed through a sigmoid
    # because we use `from_logits=True` in the loss function for better numerical stability.
    output = layers.Dense(1)(x)

    return Model([image_input, label_input], output, name="Discriminator")

# --- Create models and optimizers ---
generator = build_generator()
discriminator = build_discriminator()

generator.summary()
discriminator.summary()

generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1)

# The loss function for a binary classification problem (real vs. fake).
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# =================================================================================
# SECTION 4: TRAINING LOOP
# =================================================================================

# --- Loss Functions ---
def discriminator_loss(real_output, fake_output):
    # The discriminator wants to predict 1 for real images and 0 for fake images.
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    # The generator wants the discriminator to predict 1 for its fake images.
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# --- The Core Training Step ---
# `tf.function` compiles this function into a high-performance TensorFlow graph.
@tf.function
def train_step(images, labels):
    # Generate random noise for the generator.
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])

    # Use tf.GradientTape to record operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 1. Generate fake images.
        generated_images = generator([noise, labels], training=True)

        # 2. Get predictions from the discriminator for both real and fake images.
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        # 3. Calculate the losses.
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 4. Calculate gradients.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 5. Apply gradients to update the model weights.
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# --- Utility function to save sample images ---
def save_sample_images(epoch):
    """Generates and saves a grid of images for progress visualization."""
    num_samples_per_class = 4
    noise = tf.random.normal([NUM_CLASSES * num_samples_per_class, LATENT_DIM])
    # Create a label for each sample to generate one of each class.
    sampled_labels_np = np.array([i for i in range(NUM_CLASSES) for _ in range(num_samples_per_class)])
    
    # ✅ FIX: Use the .predict() method for inference, which is more stable than a direct call.
    # It handles the graph/eager mode switching more robustly.
    predictions = generator.predict([noise, sampled_labels_np])
    
    # Rescale images from [-1, 1] back to [0, 255] for saving.
    # The output of .predict() is already a NumPy array.
    predictions = (predictions * 127.5 + 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(NUM_CLASSES, num_samples_per_class, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
        # Add class titles to the first row.
        if i < num_samples_per_class:
            plt.title(CLASSES[i])

    plt.savefig(os.path.join(OUTPUT_DIR, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# --- The Main Training Loop ---
print("▶️ Starting GAN training...")
for epoch in range(EPOCHS):
    start_time = time.time()
    total_gen_loss = 0
    total_disc_loss = 0
    num_batches = 0

    # Iterate over each batch in the dataset.
    for image_batch, label_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch, label_batch)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1

    # Calculate and print average losses for the epoch.
    avg_gen_loss = total_gen_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches
    
    print(f'Time for epoch {epoch + 1} is {time.time()-start_time:.2f} sec | Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')

    # Save a grid of sample images every 5 epochs to check progress.
    if (epoch + 1) % 5 == 0:
        save_sample_images(epoch + 1)

# --- Save the Final Trained Generator ---
print(f"\n✅ Training complete. Saving generator model to {GENERATOR_SAVE_PATH}")
generator.save(GENERATOR_SAVE_PATH)
print("✅ Generator model saved successfully.")
