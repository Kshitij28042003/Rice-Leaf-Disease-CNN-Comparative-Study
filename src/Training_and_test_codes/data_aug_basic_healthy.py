import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 📁 Input: original images folder
input_root = r"D:\Data_sets\CP_DATASET\HEALTHY\og"

# 📁 Output: ONE folder for all augmented images
output_root = r"D:\Data_sets\CP_DATASET\HEALTHY\augmented"

# Resize target
TARGET_SIZE = (224, 224)

# Geometric aug + standard normalization
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

VALID_EXT = ('.jpg', '.jpeg', '.png')

# Create output folder if it doesn't exist
os.makedirs(output_root, exist_ok=True)

image_files = [f for f in os.listdir(input_root) if f.lower().endswith(VALID_EXT)]

if not image_files:
    print("[WARNING] No valid images found in input folder!")
else:
    total_aug = 0
    for img_name in image_files:
        img_path = os.path.join(input_root, img_name)
        try:
            img = load_img(img_path, target_size=TARGET_SIZE)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}")
            continue

        img_prefix = os.path.splitext(img_name)[0]

        print(f"[INFO] Augmenting: {img_name} -> saving to {output_root}")

        aug_iter = datagen.flow(
            x,
            batch_size=1,
            save_to_dir=output_root,  # ✅ Save to the single folder
            save_prefix=img_prefix,
            save_format='jpeg'
        )

        saved = 0
        for i in range(5):
            next(aug_iter)
            saved += 1

        print(f"[DONE] Saved {saved}/5 aug images for {img_name}")
        total_aug += saved

    print(f"\n✅ Total augmented images saved: {total_aug}")
    print(f"📂 Check your folder: {output_root}")
