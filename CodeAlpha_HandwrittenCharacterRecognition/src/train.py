"""
CodeAlpha Machine Learning Internship - Task 3: Handwritten Character Recognition
==================================================================================

What this script does, in plain words:
    It trains a Convolutional Neural Network (CNN) to read handwritten capital
    or small letters (A to Z) from small 28 x 28 pixel grayscale pictures, using
    the EMNIST Letters dataset. Then it tests the network on letters it has
    never seen and saves charts that show how well it did.

How to run it (from the project's top folder):
    python src/train.py              # full training (about 15 epochs max)
    python src/train.py --epochs 3   # quicker run, slightly lower accuracy
    python src/train.py --dataset digits   # warm-up: MNIST digits 0-9 (11 MB download)

    Tip: Google Colab with a free GPU is much faster than a laptop CPU.
    In Colab: Runtime > Change runtime type > T4 GPU, then run the notebook.

What it creates:
    images/  -> sample_grid.png, training_curves.png, confusion_matrix.png,
                misclassified.png
    results/ -> emnist_cnn_metrics.txt (test accuracy + per-letter report)
    models/  -> emnist_cnn.keras (the trained network) + emnist_cnn_classes.txt

Quick vocabulary (more in the README):
    pixel    = one dot of the image; here a number from 0 (black) to 255 (white)
    CNN      = a neural network that slides small filters over the image to find
               strokes, curves and corners, then combines them into letters
    epoch    = one full pass through all training images
    loss     = the "error" number training tries to shrink
    accuracy = share of letters the model reads correctly

The "# %%" lines split the file into cells. The notebook in notebooks/ is
built from these same cells.
"""

# %% [markdown]
# ## 1. Import the tools (libraries) we need
# - tensorflow / keras: builds and trains neural networks
# - tensorflow_datasets (tfds): downloads EMNIST for us (about 536 MB, once)
# - numpy: math on arrays of numbers
# - matplotlib / seaborn: charts
# - scikit-learn: the per-letter report and confusion matrix

# %%
import argparse
import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# Settings you can change. In a notebook, edit EPOCHS here directly.
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15, help="most passes through the data")
parser.add_argument("--dataset", choices=["letters", "digits"], default="letters",
                    help="letters = EMNIST A-Z (the real task); digits = MNIST 0-9 (quick warm-up)")
args, _ = parser.parse_known_args(sys.argv[1:] if "ipykernel" not in sys.modules else [])
EPOCHS = args.epochs
DATASET = args.dataset  # in a notebook, set this to "digits" for the warm-up
BATCH_SIZE = 128  # how many images the network looks at before each small update
SEED = 42  # makes random steps repeatable, so reruns give similar numbers

# The names of the classes the network chooses between. Index 0 = first name.
CLASSES = list(string.ascii_uppercase) if DATASET == "letters" else list(string.digits)
MODEL_NAME = "emnist_cnn" if DATASET == "letters" else "mnist_cnn"

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # in a notebook there is no __file__
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
IMAGES, RESULTS, MODELS = ROOT / "images", ROOT / "results", ROOT / "models"
for folder in (IMAGES, RESULTS, MODELS):
    folder.mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)
print("TensorFlow", tf.__version__, "| GPU found:", bool(tf.config.list_physical_devices("GPU")))

# %% [markdown]
# ## 2. Load the data
# **letters (the real task):** tfds.load downloads EMNIST Letters the first time
# (about 536 MB, a few minutes), then reuses the saved copy. 88,800 training and
# 14,800 test images, each 28 x 28 pixels, as (image, label) pairs.
#
# **digits (warm-up):** MNIST, built into Keras: 60,000 training and 10,000 test
# images of the digits 0-9. Use it to check your setup quickly.

# %%
def to_numpy(ds):
    """Turn a TensorFlow dataset into two plain numpy arrays: images and labels."""
    images, labels = [], []
    for image_batch, label_batch in tfds.as_numpy(ds.batch(10_000)):
        images.append(image_batch)
        labels.append(label_batch)
    return np.concatenate(images), np.concatenate(labels)

if DATASET == "letters":
    (train_ds, test_ds) = tfds.load("emnist/letters", split=["train", "test"], as_supervised=True)
    x_train, y_train = to_numpy(train_ds)
    x_test, y_test = to_numpy(test_ds)
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., None], x_test[..., None]  # add the 1 "channel" axis: (N, 28, 28, 1)
print("Train images:", x_train.shape, "| Test images:", x_test.shape)
print("Label values found:", np.unique(y_train))

# %% [markdown]
# ## 3. Clean the data
# Three fixes (1 and 3 are only needed for EMNIST letters):
# 1. **Orientation.** EMNIST images are stored mirrored and rotated. Swapping
#    rows and columns (a "transpose") makes them upright, as the TensorFlow
#    Datasets page suggests.
# 2. **Scale.** Pixels go from 0-255 to 0-1. Neural networks learn more
#    smoothly with small numbers.
# 3. **Labels.** EMNIST numbers letters 1 to 26 (A = 1). We shift them to
#    0 to 25 because Keras expects classes to start at 0.

# %%
if DATASET == "letters":
    x_train = np.transpose(x_train, (0, 2, 1, 3))  # fix orientation
    x_test = np.transpose(x_test, (0, 2, 1, 3))
    offset = int(y_train.min())  # EMNIST letters start at 1 (A = 1); this is usually 1
    y_train, y_test = y_train - offset, y_test - offset  # 1-26 becomes 0-25
x_train = x_train.astype("float32") / 255.0  # pixels 0-255 become 0-1
x_test = x_test.astype("float32") / 255.0
y_train, y_test = y_train.astype("int64"), y_test.astype("int64")
assert y_train.min() == 0 and y_train.max() == len(CLASSES) - 1, "labels should run 0 to number of classes - 1"

# %% [markdown]
# ## 4. Look at the data
# Always check a few images with their labels. If letters look sideways or the
# labels don't match, step 3 went wrong.

# %%
fig, axes = plt.subplots(5, 5, figsize=(6, 6))
for ax, i in zip(axes.flat, np.random.choice(len(x_train), 25, replace=False)):
    ax.imshow(x_train[i].squeeze(), cmap="gray")
    ax.set_title(CLASSES[y_train[i]], fontsize=10)
    ax.axis("off")
plt.suptitle("25 random training images and their labels")
plt.tight_layout()
plt.savefig(IMAGES / "sample_grid.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Build the CNN
# Layer by layer:
# - **RandomRotation / RandomTranslation / RandomZoom** ("data augmentation"):
#   during training only, slightly tilt, shift and zoom each image, so the
#   network learns letters, not exact pixel positions. No flips: a flipped
#   letter can be a different letter.
# - **Conv2D(32, 3)**: 32 small 3 x 3 filters slide over the image, each
#   learning to spot a pattern such as an edge. "relu" keeps positive signals.
# - **BatchNormalization**: keeps the numbers flowing between layers in a steady
#   range, which speeds up training.
# - **MaxPooling2D**: shrinks the image by half, keeping the strongest signals.
# - **Dropout**: randomly switches off some units while training, so the network
#   can't just memorize (this fights "overfitting").
# - **Flatten + Dense(256)**: turn the feature maps into one list and mix them.
# - **Dense(26, softmax)**: 26 outputs, one per letter (10 for digits), as
#   probabilities that add to 1.

# %%
def build_cnn():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation="softmax"),  # one output per class
    ])
    # adam = a popular way to adjust the weights; the loss measures how wrong the
    # probabilities are; "sparse" means our labels are plain numbers 0-25.
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_cnn()
model.summary()

# %% [markdown]
# ## 6. Train
# - validation_split=0.1 sets aside 10% of training images to check progress
#   after each epoch (the test set stays hidden until the very end).
# - EarlyStopping stops when the validation loss hasn't improved for 3 epochs
#   and rolls back to the best version.
# - ReduceLROnPlateau takes smaller learning steps when progress stalls.

# %%
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

# %% [markdown]
# ## 7. Training curves
# If the training line keeps improving while the validation line gets worse,
# the model is overfitting. Lines that move together are healthy.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs_run = range(1, len(history.history["loss"]) + 1)
ax1.plot(epochs_run, history.history["loss"], label="training")
ax1.plot(epochs_run, history.history["val_loss"], label="validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (cross-entropy)")
ax1.set_title("Loss")
ax1.legend()
ax2.plot(epochs_run, np.array(history.history["accuracy"]) * 100, label="training")
ax2.plot(epochs_run, np.array(history.history["val_accuracy"]) * 100, label="validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy")
ax2.legend()
plt.tight_layout()
plt.savefig(IMAGES / "training_curves.png", dpi=150)
plt.close()

# %% [markdown]
# ## 8. Final test on the hidden test images
# - Accuracy: share of letters read correctly.
# - Per-letter precision / recall / F1: which letters are easy and which are hard.

# %%
probabilities = model.predict(x_test, batch_size=512, verbose=0)
y_pred = probabilities.argmax(axis=1)  # pick the letter with the highest probability
test_accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=CLASSES, digits=3)
print(f"Test accuracy: {test_accuracy:.2%}")
print(report)
(RESULTS / f"{MODEL_NAME}_metrics.txt").write_text(
    f"Test accuracy: {test_accuracy:.4f}\nEpochs run: {len(epochs_run)}\n\n{report}"
)

# %% [markdown]
# ## 9. Confusion matrix and mistakes
# The confusion matrix shows, for each true letter (row), what the model guessed
# (column). Bright squares off the diagonal are letters it mixes up, such as
# I and L, or G and Q, which can look alike even to people.

# %%
cm = confusion_matrix(y_test, y_pred, normalize="true") * 100
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar_kws={"label": "% of each true letter"})
ax.set_xlabel("Predicted letter")
ax.set_ylabel("True letter")
ax.set_title(f"Confusion matrix (test accuracy {test_accuracy:.1%})")
plt.tight_layout()
plt.savefig(IMAGES / "confusion_matrix.png", dpi=150)
plt.close()

wrong = np.where(y_pred != y_test)[0][:25]
fig, axes = plt.subplots(5, 5, figsize=(7, 7))
for ax, i in zip(axes.flat, wrong):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"true {CLASSES[y_test[i]]} / guess {CLASSES[y_pred[i]]}", fontsize=8)
    ax.axis("off")
plt.suptitle("Examples the model got wrong")
plt.tight_layout()
plt.savefig(IMAGES / "misclassified.png", dpi=150)
plt.close()

# %% [markdown]
# ## 10. Save the trained network
# predict.py loads this file to read your own handwriting.

# %%
model.save(MODELS / f"{MODEL_NAME}.keras")
(MODELS / f"{MODEL_NAME}_classes.txt").write_text("".join(CLASSES))  # e.g. "ABC...Z"
print("Saved", MODELS / f"{MODEL_NAME}.keras")
