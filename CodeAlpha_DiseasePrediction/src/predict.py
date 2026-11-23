"""
Use the saved model to predict one patient.

How to run (after src/train.py has created models/best_model.joblib):
    python src/predict.py            # predicts a few patients from the dataset
    python src/predict.py 25         # predicts patient number 25 (0 to 568)

In a real hospital tool you would type in a new patient's 30 measurements.
Here we borrow rows from the dataset so you can see it work. This is a
learning project, not a medical device.
"""

import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent.parent
saved = joblib.load(ROOT / "models" / "best_model.joblib")
model, features = saved["model"], saved["features"]

data = load_breast_cancer(as_frame=True)
X = data.data[features]
truth = 1 - data.target  # same flip as in train.py: 1 = malignant

# Which patients to show: the number typed after the script name, or three examples.
rows = [int(a) for a in sys.argv[1:]] or [0, 19, 100]

print(f"Model: {saved['name']}\n")
for i in rows:
    patient = X.iloc[[i]]  # double brackets keep it as a one-row table
    p_malignant = model.predict_proba(patient)[0, 1]
    guess = "Malignant" if p_malignant >= 0.5 else "Benign"
    actual = "Malignant" if truth.iloc[i] == 1 else "Benign"
    print(f"Patient {i:3d}: predicted {guess:9s} (chance malignant {p_malignant:6.1%}) | actual {actual}")
