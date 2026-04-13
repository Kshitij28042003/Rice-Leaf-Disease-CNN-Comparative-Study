# =================================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =================================================================================
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
import time
import matplotlib.pyplot as plt
from PIL import Image

print(f"TensorFlow Version: {tf.__version__}")

# --- Performance Optimizations ---
# ✅ NEW: Enable Mixed Precision Training for significant speedup on compatible GPUs.
# This uses a mix of 16-bit and 32-bit precision for calculations.
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision training enabled.")

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
CLASS_TO_TRAIN = "BLIGHT" 
BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET"

# --- GAN & Image Parameters ---
IMG_SIZE = 128 
CHANNELS = 3
LATENT_DIM = 128
BATCH_SIZE = 32

# --- Training Parameters ---
EPOCHS = 300
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
SAVE_CHECKPOINT_FREQ = 20
N_CRITIC = 5
GP_WEIGHT = 10.0

# --- Output Paths ---
OUTPUT_DIR = f"gan_wgangp_final_{CLASS_TO_TRAIN.lower()}_output" # New folder
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


# =================================================================================
# SECTION 2: DATA LOADING
# =================================================================================
print(f"▶️ Loading dataset for class: {CLASS_TO_TRAIN}...")
def load_dataset():
    class_folder = os.path.join(BASE_PATH, CLASS_TO_TRAIN, "augmented_for_gan")
    files = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.jpeg")) + glob.glob(os.path.join(class_folder, "*.JPG"))
    
    if not files:
        print(f"❌ ERROR: No images found in {class_folder}. Please check the path.")
        return None

    def process_img(filepath):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(files)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Dataset loaded with {len(files)} images.")
    return dataset

train_dataset = load_dataset()

if train_dataset is None:
    exit()

# =================================================================================
# SECTION 3: WGAN-GP ARCHITECTURE & AUGMENTATION
# =================================================================================
# ✅ NEW: Added more challenging augmentations to make the critic's job harder.
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# Deeper Generator
def build_generator():
    model = Sequential(name="Generator")
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 512)))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # The final layer uses float32 for numerical stability, as recommended for mixed precision.
    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', activation='tanh', dtype='float32'))
    return model

# Deeper Critic
def build_critic():
    model = Sequential(name="Critic")
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    # The final layer uses float32 for numerical stability.
    model.add(layers.Dense(1, dtype='float32'))
    return model

# --- Build the models ---
print("▶️ Building new DEEPER WGAN-GP models from scratch...")
generator = build_generator()
critic = build_critic()
print("✅ New models built.")

# --- Create optimizers ---
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)

# =================================================================================
# SECTION 4: WGAN-GP TRAINING LOOP WITH AUGMENTATION
# =================================================================================
def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def gradient_penalty(real_images, fake_images):
    # ✅ FIX: Explicitly cast all inputs to float32 to prevent type mismatch errors
    # that can occur with mixed precision training.
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    
    alpha = tf.random.uniform([tf.shape(real_images)[0], 1, 1, 1], 0., 1.)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        pred = critic(interpolated_images, training=True)
        # Also cast prediction to float32 for stability
        pred = tf.cast(pred, tf.float32)

    grads = tape.gradient(pred, [interpolated_images])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])
    
    for _ in range(N_CRITIC):
        with tf.GradientTape() as tape:
            augmented_images = data_augmentation(images, training=True)
            
            fake_images = generator(noise, training=True)
            real_output = critic(augmented_images, training=True)
            fake_output = critic(fake_images, training=True)
            
            c_loss = critic_loss(real_output, fake_output)
            gp = gradient_penalty(augmented_images, fake_images)
            total_critic_loss = c_loss + gp * GP_WEIGHT
        
        critic_grad = tape.gradient(total_critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))

    with tf.GradientTape() as tape:
        generated_images = generator(noise, training=True)
        fake_output = critic(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
    
    gen_grad = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
    
    return gen_loss, total_critic_loss

# --- Utility to save sample images ---
fixed_seed_noise = tf.random.normal([16, LATENT_DIM])
def save_sample_images(epoch):
    predictions = generator(fixed_seed_noise, training=False)
    # Ensure predictions are float32 before math operations
    predictions = tf.cast(predictions, tf.float32)
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
    fig = plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
    plt.suptitle(f"Generated {CLASS_TO_TRAIN} at Epoch {epoch}", fontsize=16)
    save_path = os.path.join(OUTPUT_DIR, f'{CLASS_TO_TRAIN.lower()}_sample_epoch_{epoch:04d}.png')
    plt.savefig(save_path)
    plt.close()

# --- The Main Training Loop ---
print(f"▶️ Starting WGAN-GP training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    start_time = time.time()
    total_gen_loss, total_critic_loss, num_batches = 0, 0, 0
    for image_batch in train_dataset:
        gen_loss, critic_loss_val = train_step(image_batch)
        total_gen_loss += gen_loss
        total_critic_loss += critic_loss_val
        num_batches += 1
    avg_gen_loss = total_gen_loss / num_batches
    avg_critic_loss = total_critic_loss / num_batches
    print(f'Time for epoch {epoch + 1}/{EPOCHS} is {time.time()-start_time:.2f} sec | Gen Loss: {avg_gen_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}')
    
    if (epoch + 1) % 5 == 0:
        save_sample_images(epoch + 1)
        
    if (epoch + 1) % SAVE_CHECKPOINT_FREQ == 0:
        gen_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{CLASS_TO_TRAIN.lower()}_generator_epoch_{epoch+1:04d}.h5")
        critic_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{CLASS_TO_TRAIN.lower()}_critic_epoch_{epoch+1:04d}.h5")
        generator.save(gen_ckpt_path)
        critic.save(critic_ckpt_path)
        print(f"✅ Saved checkpoint for epoch {epoch + 1}")

# =================================================================================
# SECTION 5: SAVE THE FINAL MODELS
# =================================================================================
print(f"\n✅ Training complete. Saving final models...")
generator.save(os.path.join(OUTPUT_DIR, f"{CLASS_TO_TRAIN.lower()}_generator_final.h5"))
critic.save(os.path.join(OUTPUT_DIR, f"{CLASS_TO_TRAIN.lower()}_critic_final.h5"))
print("✅ Final models saved successfully.")
