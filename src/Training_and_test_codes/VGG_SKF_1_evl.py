import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === CONFIG ===
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_PATH = r"D:\Data_sets\CP_DATASET"
MODEL_DIR = r"D:\Data_sets\CP_MODEL"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Lower batch size for safe GPU use
N_FOLDS = 5

# === Load ALL test data ===
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
print(f"✅ Total images for final ensemble eval: {len(all_filepaths)}")

# === Preprocessing ===
def process_img(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, depth=len(CLASSES))
    return img, label

test_ds = tf.data.Dataset.from_tensor_slices((all_filepaths, all_labels))
test_ds = test_ds.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Ensemble predict ===
all_probs = []

for fold in range(1, N_FOLDS + 1):
    model_path = os.path.join(MODEL_DIR, f"FOLD_{fold}_VGG_PHASE1.h5")
    print(f"🔗 Loading model: {model_path}")
    model = load_model(model_path)

    # Predict once on the whole dataset
    probs = model.predict(test_ds, verbose=0)
    all_probs.append(probs)

    # Free up GPU memory
    tf.keras.backend.clear_session()

# === Average ensemble softmax ===
avg_probs = np.mean(all_probs, axis=0)
y_pred = np.argmax(avg_probs, axis=1)
y_true = all_labels

print("\n✅ Ensemble Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - 5-Fold Ensemble (Phase 1)")
plt.tight_layout()
plt.show()
