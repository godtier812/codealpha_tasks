"""
Read a letter from your own photo with the trained network.

How to run (after src/train.py has created models/emnist_cnn.keras):
    python src/predict.py samples/my_letter.png
    python src/predict.py samples/*.png          # several photos at once
    python src/predict.py --digits samples/my_digit.png   # use the MNIST warm-up model

How to make a good photo:
    - Write ONE capital or small letter with a thick dark marker on white paper.
    - Take a photo or screenshot and crop it so the letter fills most of the picture.
    - Save it as .png or .jpg in the samples/ folder.

What the script does to your photo, step by step:
    1. Opens it in grayscale (shades of gray, no color).
    2. Inverts it: EMNIST letters are WHITE on BLACK, while paper is the opposite.
    3. Crops to the letter, pads it to a square and shrinks it to 28 x 28 pixels,
       like the training images (the letter sits in the middle with a small border).
    4. Asks the network for probabilities and prints the top 3 guesses.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent


def prepare(path):
    """Turn a photo of one dark letter on light paper into a 28 x 28 EMNIST-style array."""
    img = ImageOps.invert(Image.open(path).convert("L"))  # white letter on black
    arr = np.array(img, dtype="float32")
    arr[arr < 60] = 0  # drop faint paper texture and shadows

    rows, cols = np.nonzero(arr)  # where the ink is
    if len(rows) == 0:
        raise ValueError(f"No ink found in {path}. Is the letter dark on light paper?")
    arr = arr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]  # crop to the letter

    side = max(arr.shape)  # pad to a square so the letter isn't stretched
    square = np.zeros((side, side), dtype="float32")
    top, left = (side - arr.shape[0]) // 2, (side - arr.shape[1]) // 2
    square[top: top + arr.shape[0], left: left + arr.shape[1]] = arr

    letter = Image.fromarray(square.astype("uint8")).resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)  # 4-pixel border, like EMNIST
    canvas.paste(letter, (4, 4))
    return np.array(canvas, dtype="float32")[None, :, :, None] / 255.0  # shape (1, 28, 28, 1)


def main():
    args = sys.argv[1:]
    name = "mnist_cnn" if "--digits" in args else "emnist_cnn"
    paths = [a for a in args if a != "--digits"]
    if not paths:
        print(__doc__)
        return
    model = keras.models.load_model(ROOT / "models" / f"{name}.keras")
    classes = (ROOT / "models" / f"{name}_classes.txt").read_text().strip()  # e.g. "ABC...Z"
    for path in paths:
        probs = model.predict(prepare(path), verbose=0)[0]
        top3 = probs.argsort()[::-1][:3]
        guesses = ", ".join(f"{classes[i]} ({probs[i]:.0%})" for i in top3)
        print(f"{Path(path).name}: {guesses}")


if __name__ == "__main__":
    main()
