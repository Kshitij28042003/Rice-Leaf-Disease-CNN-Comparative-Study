import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_PATH = r"D:\Data_sets\CP_DATASET"
MODEL_DIR = r"D:\Data_sets\CP_MODEL"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
N_FOLDS = 5

# === Load filepaths & labels ===
all_filepaths, all_labels = [], []
for idx, class_name in enumerate(CLASSES):
    aug_path = os.path.join(BASE_PATH, class_name, "augmented")
    files = glob.glob(os.path.join(aug_path, "*.jpg")) + \
            glob.glob(os.path.join(aug_path, "*.jpeg")) + \
            glob.glob(os.path.join(aug_path, "*.png"))
    all_filepaths.extend(files)
    all_labels.extend([idx] * len(files))

all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)
print(f"✅ Total images: {len(all_filepaths)}")

# === Color Jitter ===
def color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    return tf.clip_by_value(image, 0.0, 1.0)

# === GridMask ===
def apply_gridmask(image, d_min=10, d_max=40, rotate=False, ratio=0.6):
    d = tf.random.uniform([], d_min, d_max, dtype=tf.int32)
    l = tf.cast(tf.cast(d, tf.float32) * ratio, tf.int32)

    mask = tf.ones(IMG_SIZE, dtype=tf.float32)
    for i in range(0, IMG_SIZE[0], d):
        for j in range(0, IMG_SIZE[1], d):
            y = i + tf.random.uniform([], 0, d - l, dtype=tf.int32)
            x = j + tf.random.uniform([], 0, d - l, dtype=tf.int32)
            y1 = tf.clip_by_value(y, 0, IMG_SIZE[0])
            y2 = tf.clip_by_value(y + l, 0, IMG_SIZE[0])
            x1 = tf.clip_by_value(x, 0, IMG_SIZE[1])
            x2 = tf.clip_by_value(x + l, 0, IMG_SIZE[1])
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.reshape(tf.stack(tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing='ij'), -1), [-1, 2]),
                tf.zeros([(y2 - y1) * (x2 - x1)], dtype=tf.float32)
            )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    return image * mask

# === Image processor ===
def process_img(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = color_jitter(img)
    img = apply_gridmask(img)
    label = tf.one_hot(label, depth=len(CLASSES))
    return img, label

# === Safe CutMix ===
def sample_beta_distribution(alpha, shape):
    gamma1 = tf.random.gamma(shape, alpha)
    gamma2 = tf.random.gamma(shape, alpha)
    return gamma1 / (gamma1 + gamma2)

def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    lam = sample_beta_distribution(alpha, [batch_size])
    lam = tf.cast(lam, tf.float32)

    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    def _cutmix_single(args):
        image, label, shuf_image, shuf_label, lam_scalar = args

        cut_rat = tf.math.sqrt(1. - lam_scalar)
        cut_w = tf.cast(IMG_SIZE[1] * cut_rat, tf.int32)
        cut_h = tf.cast(IMG_SIZE[0] * cut_rat, tf.int32)

        cx = tf.random.uniform((), 0, IMG_SIZE[1], dtype=tf.int32)
        cy = tf.random.uniform((), 0, IMG_SIZE[0], dtype=tf.int32)

        x1 = tf.clip_by_value(cx - cut_w // 2, 0, IMG_SIZE[1])
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, IMG_SIZE[0])
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, IMG_SIZE[1])
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, IMG_SIZE[0])

        # Replace region
        patch = shuf_image[y1:y2, x1:x2, :]
        paddings = [
            [y1, IMG_SIZE[0] - y2],
            [x1, IMG_SIZE[1] - x2],
            [0, 0]
        ]
        mask = tf.pad(tf.ones_like(patch), paddings, constant_values=0)
        new_image = image * (1 - mask) + tf.pad(patch, paddings, constant_values=0)

        new_label = label * lam_scalar + shuf_label * (1. - lam_scalar)
        return new_image, new_label

    images, labels = tf.map_fn(
        _cutmix_single,
        (images, labels, shuffled_images, shuffled_labels, lam),
        dtype=(images.dtype, labels.dtype)
    )
    return images, labels

# === LR Logger ===
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, '__call__'):
            lr = lr(self.model.optimizer.iterations)
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        print(f"📉 Learning rate at epoch {epoch+1}: {lr:.6f}")

# === Fine-tuning loop ===
for fold in range(1, N_FOLDS + 1):
    print(f"\n🚀 Phase 2 Fine-tune + CutMix + GridMask Fold {fold}/{N_FOLDS}")

    train_idx = np.load(f"fold_{fold}_train_idx.npy")
    val_idx = np.load(f"fold_{fold}_val_idx.npy")
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

    BUFFER_SIZE = len(train_idx)

    train_ds = tf.data.Dataset.from_tensor_slices((all_filepaths[train_idx], train_labels))
    train_ds = train_ds.shuffle(BUFFER_SIZE, seed=SEED)\
                       .map(process_img, num_parallel_calls=tf.data.AUTOTUNE)\
                       .batch(BATCH_SIZE)\
                       .map(lambda x, y: cutmix(x, y, alpha=1.0), num_parallel_calls=tf.data.AUTOTUNE)\
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((all_filepaths[val_idx], val_labels))
    val_ds = val_ds.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)\
                   .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    class_weights_fold = dict(enumerate(
        class_weight.compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=train_labels)
    ))

    print(f"✅ Class weights: {class_weights_fold}")

    phase1_model_path = os.path.join(MODEL_DIR, f"FOLD_{fold}_VGG_PHASE1.h5")
    model = load_model(phase1_model_path)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            LearningRateLogger()
        ],
        class_weight=class_weights_fold
    )

    # === Eval ===
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("\n📊 Fold Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - Phase 2 CutMix+GridMask Fold {fold}")
    plt.tight_layout()
    plt.show()

    SAVE_PATH = os.path.join(MODEL_DIR, f"FOLD_{fold}_VGG_P2_SKF.h5")
    model.save(SAVE_PATH)
    print(f"✅ Phase 2 CutMix+GridMask Model saved: {SAVE_PATH}")
