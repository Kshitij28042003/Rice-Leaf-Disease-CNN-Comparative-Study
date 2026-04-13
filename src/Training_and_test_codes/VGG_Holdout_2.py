# ============================================
# ✅ PHASE 1: VGG16 BASE TRAINING
# ============================================

import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === CONFIG ===
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_DATASET_F1"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60

# === Load filepaths & labels ===
all_filepaths, all_labels = [], []
for idx, class_name in enumerate(CLASSES):
    aug_path = os.path.join(BASE_PATH, class_name, "augmented")
    files = glob.glob(os.path.join(aug_path, "*.jpg")) + \
            glob.glob(os.path.join(aug_path, "*.jpeg")) + \
            glob.glob(os.path.join(aug_path, "*.png"))
    all_filepaths.extend(files)
    all_labels.extend([idx] * len(files))

print(f"✅ Total images found: {len(all_filepaths)}")

# === Dataset ===
filepaths_ds = tf.data.Dataset.from_tensor_slices(all_filepaths)
labels_ds = tf.data.Dataset.from_tensor_slices(all_labels)
ds = tf.data.Dataset.zip((filepaths_ds, labels_ds)).shuffle(len(all_filepaths), seed=SEED)

train_size = int(0.8 * len(all_filepaths))
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

# === Color Jitter ===
def color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    return tf.clip_by_value(image, 0.0, 255.0)  # VGG16 wants [0, 255] before preprocess

# === GridMask ===
def grid_mask(image, d_min=50, d_max=100, ratio=0.5):
    h, w, _ = image.shape
    d = tf.random.uniform([], d_min, d_max, dtype=tf.int32)
    l = tf.cast(tf.cast(d, tf.float32) * ratio, tf.int32)

    mask = tf.ones([h, w], dtype=tf.float32)
    for i in range(0, h, d):
        for j in range(0, w, d):
            y1 = i
            y2 = tf.minimum(i + l, h)
            x1 = j
            x2 = tf.minimum(j + l, w)

            y_range = tf.range(y1, y2)
            x_range = tf.range(x1, x2)
            yy, xx = tf.meshgrid(y_range, x_range, indexing='ij')
            indices = tf.stack([yy, xx], axis=-1)
            indices = tf.reshape(indices, [-1, 2])

            mask = tf.tensor_scatter_nd_update(
                mask,
                indices,
                tf.zeros([(y2 - y1) * (x2 - x1)], dtype=tf.float32)
            )

    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, [1, 1, 3])
    return image * mask

# === Image processor ===
def process_img(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)

    img = color_jitter(img)
    img = grid_mask(img)

    img = preprocess_input(img)  # Correct for VGG16!
    label = tf.one_hot(label, depth=len(CLASSES))
    return img, label

# === CutMix ===
def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size])

    rand_idx = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, rand_idx)
    labels2 = tf.gather(labels, rand_idx)

    cut_rat = tf.math.sqrt(1. - lam)
    cut_w = tf.cast(img_w, tf.float32) * cut_rat
    cut_h = tf.cast(img_h, tf.float32) * cut_rat

    cx = tf.random.uniform([batch_size], 0, tf.cast(img_w, tf.float32))
    cy = tf.random.uniform([batch_size], 0, tf.cast(img_h, tf.float32))

    x1 = tf.cast(cx - cut_w / 2, tf.int32)
    y1 = tf.cast(cy - cut_h / 2, tf.int32)
    x2 = tf.cast(cx + cut_w / 2, tf.int32)
    y2 = tf.cast(cy + cut_h / 2, tf.int32)

    x1 = tf.clip_by_value(x1, 0, img_w)
    y1 = tf.clip_by_value(y1, 0, img_h)
    x2 = tf.clip_by_value(x2, 0, img_w)
    y2 = tf.clip_by_value(y2, 0, img_h)

    def apply_cutmix(i):
        img1 = images[i]
        img2 = images2[i]
        bbx1, bby1, bbx2, bby2 = x1[i], y1[i], x2[i], y2[i]

        mask = tf.pad(
            tf.zeros([bby2 - bby1, bbx2 - bbx1, 3]),
            [[bby1, img_h - bby2], [bbx1, img_w - bbx2], [0, 0]],
            constant_values=1.0
        )
        mask = 1.0 - mask
        mixed = img1 * mask + img2 * (1.0 - mask)

        area = tf.cast(bbx2 - bbx1, tf.float32) * tf.cast(bby2 - bby1, tf.float32)
        lam_adjusted = 1.0 - (area / tf.cast(img_w * img_h, tf.float32))
        new_label = lam_adjusted * labels[i] + (1.0 - lam_adjusted) * labels2[i]

        return mixed, new_label

    mixed_images, mixed_labels = tf.map_fn(
        apply_cutmix,
        tf.range(batch_size),
        fn_output_signature=(tf.float32, tf.float32)
    )

    return mixed_images, mixed_labels

# === Final dataset ===
train_ds = train_ds.map(process_img).batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(process_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === VGG16 model ===
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Reload dataset & pipeline same as Phase 1
# (Reuse `process_img`, `cutmix`, etc.)

model = load_model("/mnt/shared/College_work_data/Data_sets/CP_MODEL/VGG16_PHASE1_F1.h5")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, '__call__'): lr = lr(self.model.optimizer.iterations)
        if hasattr(lr, 'numpy'): lr = lr.numpy()
        print(f"📉 Learning rate at epoch {epoch+1}: {lr:.6f}")

# === Class weights ===
y_train_int = np.argmax(np.concatenate([labels.numpy() for _, labels in train_ds.unbatch().batch(BATCH_SIZE)]), axis=1)
class_weights = dict(enumerate(class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(CLASSES)),
    y=y_train_int
)))
print("✅ Computed class weights:", class_weights)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=60,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True), LearningRateLogger()],
    class_weight=class_weights
)

# === Evaluate ===
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

print(classification_report(y_true, y_pred, target_names=CLASSES))
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.show()

SAVE_PATH = "/mnt/shared/College_work_data/Data_sets/CP_MODEL/VGG16_PHASE2_F1.h5"
model.save(SAVE_PATH)
print(f"✅ Model saved at: {SAVE_PATH}")
print("✅ Phase 2 fine-tuned model saved.")
