import os
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report

src_root = "."
filtered_root = "filtered_dataset"
split_root = "split_dataset"

classes = ["buffalo", "elephant", "rhino", "zebra"]
image_exts = {".jpg"}

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

img_size = (128, 128)
batch_size = 32
epochs = 15

random.seed(42)

# =========================
# filtering
# =========================

if os.path.isdir(filtered_root):
    print("Dataset is already filtered")
else:
    os.makedirs(filtered_root, exist_ok=True)

    for class_name in classes:
        src_dir = os.path.join(src_root, class_name)
        dst_dir = os.path.join(filtered_root, class_name)

        if not os.path.isdir(src_dir):
            print(f"missing folder: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        for name in sorted(os.listdir(src_dir)):
            src_file = os.path.join(src_dir, name)

            if not os.path.isfile(src_file):
                continue

            ext = os.path.splitext(name)[1].lower()

            if ext not in image_exts:
                continue

            new_name = os.path.splitext(name)[0] + ext
            dst_file = os.path.join(dst_dir, new_name)

            if os.path.exists(dst_file):
                continue

            shutil.copy2(src_file, dst_file)

    print("Filtering finished")

# =========================
# filtered dataset statistics
# =========================

print("\nFiltered dataset statistics:")

filtered_total = 0

for class_name in classes:
    class_dir = os.path.join(filtered_root, class_name)

    if not os.path.isdir(class_dir):
        print(f"{class_name}: folder not found")
        continue

    count = 0

    for name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, name)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext not in image_exts:
            continue

        count += 1

    filtered_total += count
    print(f"{class_name}: {count} images")

print(f"Total: {filtered_total} images")

# =========================
# split
# =========================

if os.path.isdir(split_root):
    print("\nDataset is already split")
else:
    for split_name in ["train", "val", "test"]:
        for class_name in classes:
            os.makedirs(os.path.join(split_root, split_name, class_name), exist_ok=True)

    for class_name in classes:
        src_dir = os.path.join(filtered_root, class_name)

        if not os.path.isdir(src_dir):
            print(f"missing folder: {src_dir}")
            continue

        images = []

        for name in os.listdir(src_dir):
            file_path = os.path.join(src_dir, name)

            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(name)[1].lower()

            if ext not in image_exts:
                continue

            images.append(name)

        images.sort()
        random.shuffle(images)

        total = len(images)

        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for name in train_images:
            src_file = os.path.join(src_dir, name)
            dst_file = os.path.join(split_root, "train", class_name, name)
            shutil.copy2(src_file, dst_file)

        for name in val_images:
            src_file = os.path.join(src_dir, name)
            dst_file = os.path.join(split_root, "val", class_name, name)
            shutil.copy2(src_file, dst_file)

        for name in test_images:
            src_file = os.path.join(src_dir, name)
            dst_file = os.path.join(split_root, "test", class_name, name)
            shutil.copy2(src_file, dst_file)

        print(f"{class_name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    print("Splitting finished")

# =========================
# split statistics
# =========================

print("\nSplit statistics:")

for split_name in ["train", "val", "test"]:
    print(f"\n{split_name}:")
    total_count = 0

    for class_name in classes:
        class_dir = os.path.join(split_root, split_name, class_name)

        if not os.path.isdir(class_dir):
            print(f"{class_name}: folder not found")
            continue

        count = 0

        for name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, name)

            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(name)[1].lower()

            if ext not in image_exts:
                continue

            count += 1

        total_count += count
        print(f"{class_name}: {count}")

    print(f"total {split_name}: {total_count}")

# =========================
# load datasets
# =========================

train_dir = os.path.join(split_root, "train")
val_dir = os.path.join(split_root, "val")
test_dir = os.path.join(split_root, "test")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
print("\nClasses:", class_names)

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First labels:", labels[:10].numpy())

# =========================
# model
# =========================

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_ds)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# =========================
# classification report
# =========================

y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
