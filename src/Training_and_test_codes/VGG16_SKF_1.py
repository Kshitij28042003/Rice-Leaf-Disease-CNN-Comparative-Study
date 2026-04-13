import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ✅ CONFIG
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_PATH = r"D:\Data_sets\CP_DATASET"
CLASSES = ["BLIGHT", "BLAST", "BROWNSPOT", "HEALTHY"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # You can test 32 or 16 if you get OOM
EPOCHS = 30
N_FOLDS = 5
BUFFER_SIZE = 1000  # For shuffle

# ✅ Collect filepaths & labels
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
print(f"✅ Total images found: {len(all_filepaths)}")

# ✅ Color Jitter
def color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    return tf.clip_by_value(image, 0.0, 1.0)

# ✅ Image processor
def process_img(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = color_jitter(img)
    label = tf.one_hot(label, depth=len(CLASSES))
    return img, label

# ✅ Learning rate logger
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, '__call__'):
            lr = lr(self.model.optimizer.iterations)
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        print(f"📉 Learning rate at epoch {epoch+1}: {lr:.6f}")

# ✅ Stratified KFold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels), 1):
    print(f"\n🚀 Fold {fold}/{N_FOLDS}")
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]
    print(f"Train counts: {np.bincount(train_labels)}")
    print(f"Val counts: {np.bincount(val_labels)}")

    np.save(f"fold_{fold}_train_idx.npy", train_idx)
    np.save(f"fold_{fold}_val_idx.npy", val_idx)

    # ✅ Faster dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices(
        (all_filepaths[train_idx], train_labels)
    ).shuffle(len(train_idx), seed=SEED)\
     .map(process_img, num_parallel_calls=tf.data.AUTOTUNE)\
     .batch(BATCH_SIZE)\
     .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (all_filepaths[val_idx], val_labels)
    ).map(process_img, num_parallel_calls=tf.data.AUTOTUNE)\
     .batch(BATCH_SIZE)\
     .prefetch(tf.data.AUTOTUNE)

    # ✅ Compute class weights
    class_weights_fold = dict(enumerate(
        class_weight.compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=train_labels)
    ))
    print(f"✅ Class weights: {class_weights_fold}")

    # ✅ Model: frozen VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(CLASSES), activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # ✅ Train with early stopping & logger
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            LearningRateLogger()
        ],
        class_weight=class_weights_fold
    )

    # ✅ Evaluate
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("\n📊 Fold Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.show()

    SAVE_PATH = f"D://Data_sets//CP_MODEL//FOLD_{fold}_VGG_PHASE1.h5"
    model.save(SAVE_PATH)
    print(f"✅ Model saved: {SAVE_PATH}")
