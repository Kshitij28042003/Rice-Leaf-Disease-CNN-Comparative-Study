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
from PIL import Image

print(f"TensorFlow Version: {tf.__version__}")

# Configure GPU memory growth to prevent cuBLAS initialization errors.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Enabled memory growth for {len(gpus)} GPU(s).")
  except RuntimeError as e:
    print(f"❌ Error setting memory growth: {e}")


# --- Configuration ---
BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
NUM_CLASSES = len(CLASSES)

# --- GAN & Image Parameters ---
IMG_SIZE = 128 
CHANNELS = 3
LATENT_DIM = 128
BATCH_SIZE = 32

# --- Training Parameters ---
EPOCHS = 200
GEN_LR = 0.0002
DISC_LR = 0.0001
ADAM_BETA_1 = 0.5
SAVE_CHECKPOINT_FREQ = 10

# --- Output Paths ---
OUTPUT_DIR = "gan_scratch_tuned_lr_output" 
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
AUGMENTATION_SAVE_DIR = os.path.join(OUTPUT_DIR, "periodic_augmentations")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(AUGMENTATION_SAVE_DIR):
    os.makedirs(AUGMENTATION_SAVE_DIR)


# =================================================================================
# SECTION 2: DATA LOADING & PREPARATION (No Changes)
# =================================================================================
print("▶️ Loading and preprocessing dataset...")
def load_dataset():
    all_filepaths, all_labels = [], []
    for idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(BASE_PATH, class_name, "augmented")
        files = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.jpeg")) + glob.glob(os.path.join(class_folder, "*.png"))
        all_filepaths.extend(files)
        all_labels.extend([idx] * len(files))
    
    def process_img(filepath):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    filepaths_ds = tf.data.Dataset.from_tensor_slices(all_filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int32))
    images_ds = filepaths_ds.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))
    dataset = dataset.shuffle(buffer_size=len(all_filepaths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f"✅ Dataset loaded with {len(all_filepaths)} images.")
    return dataset

train_dataset = load_dataset()

# =================================================================================
# SECTION 3: GAN ARCHITECTURE (BUILDING FROM SCRATCH)
# =================================================================================
def build_generator():
    noise_input = Input(shape=(LATENT_DIM,), name="noise_input")
    label_input = Input(shape=(1,), dtype='int32', name="label_input")
    
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(4 * 4)(label_embedding)
    label_embedding = layers.Reshape((4, 4, 1))(label_embedding)
    noise_path = layers.Dense(4 * 4 * 256, use_bias=False)(noise_input)
    noise_path = layers.BatchNormalization()(noise_path)
    noise_path = layers.LeakyReLU()(noise_path)
    noise_path = layers.Reshape((4, 4, 256))(noise_path)
    merged = layers.Concatenate()([noise_path, label_embedding])
    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(4, 4), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    output_image = layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x)
    return Model([noise_input, label_input], output_image, name="Generator")

def build_discriminator():
    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(IMG_SIZE * IMG_SIZE)(label_embedding)
    label_embedding = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_embedding)
    
    image_input = Input(shape=[IMG_SIZE, IMG_SIZE, CHANNELS])
    
    merged = layers.Concatenate()([image_input, label_embedding])

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
    output = layers.Dense(1)(x)
    return Model([image_input, label_input], output, name="Discriminator")

# --- Build the models ---
print("▶️ Building new generator and discriminator from scratch...")
generator = build_generator()
discriminator = build_discriminator()
print("✅ New models built.")

# --- Create optimizers and loss function ---
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=DISC_LR, beta_1=ADAM_BETA_1)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# =================================================================================
# SECTION 4: TRAINING LOOP
# =================================================================================
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, labels):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# ✅ NEW: A dedicated, compiled function for running generator inference.
@tf.function
def run_generator_inference(noise, labels):
    """Runs the generator in inference mode within a compiled graph."""
    return generator([noise, labels], training=False)

def save_sample_images(epoch):
    num_samples_per_class = 4
    noise = tf.random.normal([NUM_CLASSES * num_samples_per_class, LATENT_DIM])
    sampled_labels_np = np.array([i for i in range(NUM_CLASSES) for _ in range(num_samples_per_class)])
    sampled_labels_tf = tf.convert_to_tensor(sampled_labels_np, dtype=tf.int32)
    
    # ✅ FIX: Use the new compiled inference function.
    predictions = run_generator_inference(noise, sampled_labels_tf)
    
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(NUM_CLASSES, num_samples_per_class, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
        if i < num_samples_per_class:
            plt.title(CLASSES[i])
    plt.savefig(os.path.join(OUTPUT_DIR, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

def generate_and_save_for_augmentation(epoch, num_images_per_class=50):
    """Generates and saves a set of images for each class."""
    print(f"▶️ Saving augmentation images for epoch {epoch}...")
    epoch_dir = os.path.join(AUGMENTATION_SAVE_DIR, f"epoch_{epoch:04d}")
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(epoch_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        noise = tf.random.normal([num_images_per_class, LATENT_DIM])
        labels_np = np.full((num_images_per_class, 1), class_idx)
        labels_tf = tf.convert_to_tensor(labels_np, dtype=tf.int32)
        
        # ✅ FIX: Use the new compiled inference function.
        generated_images = run_generator_inference(noise, labels_tf)
        
        generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)
        
        for i, img_array in enumerate(generated_images):
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f'gan_generated_{i:04d}.png'))
    print(f"✅ Augmentation images saved for epoch {epoch}.")


# --- The Main Training Loop ---
print(f"▶️ Starting advanced training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    start_time = time.time()
    total_gen_loss, total_disc_loss, num_batches = 0, 0, 0
    for image_batch, label_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch, label_batch)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1
    avg_gen_loss = total_gen_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches
    print(f'Time for epoch {epoch + 1}/{EPOCHS} is {time.time()-start_time:.2f} sec | Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')
    
    if (epoch + 1) % 5 == 0:
        save_sample_images(epoch + 1)
        generate_and_save_for_augmentation(epoch + 1)
        
    if (epoch + 1) % SAVE_CHECKPOINT_FREQ == 0:
        gen_ckpt_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1:04d}.h5")
        disc_ckpt_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch+1:04d}.h5")
        generator.save(gen_ckpt_path)
        discriminator.save(disc_ckpt_path)
        print(f"✅ Saved checkpoint for epoch {epoch + 1}")

# =================================================================================
# SECTION 5: SAVE THE FINAL MODELS
# =================================================================================
print(f"\n✅ Training complete. Saving final models...")
generator.save(os.path.join(OUTPUT_DIR, "generator_final.h5"))
discriminator.save(os.path.join(OUTPUT_DIR, "discriminator_final.h5"))
print("✅ Generator and Discriminator models saved successfully.")
