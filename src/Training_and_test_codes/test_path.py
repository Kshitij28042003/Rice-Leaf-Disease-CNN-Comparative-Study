import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

import tensorflow as tf

# Correct import — adjust if needed
from R101_2 import CutMix, GridMask

print("A")

model_path = "/mnt/shared/College_work_data/Data_sets/CP_MODEL/ResNet101_Phase1_CutMix_GridMask.h5"

print("B")

try:
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"CutMix": CutMix, "GridMask": GridMask}
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
