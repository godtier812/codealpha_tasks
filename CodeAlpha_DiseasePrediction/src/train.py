"""
CodeAlpha Machine Learning Internship - Task 4: Disease Prediction from Medical Data
====================================================================================

What this script does, in plain words:
    It teaches four different computer models to look at 30 measurements of a
    breast tumor (taken from a microscope image) and guess whether the tumor is
    MALIGNANT (cancer) or BENIGN (not cancer). Then it tests each model on
    patients it has never seen and reports how often each one is right.

How to run it (from the project's top folder):
    python src/train.py

What it creates:
    images/   -> charts (class balance, correlation heatmap, confusion matrix,
                 ROC curves, feature importance)
    results/  -> results.csv, a table of every model's scores
    models/   -> best_model.joblib, the winning model saved to a file

Quick vocabulary (more in the README):
    feature  = one input column (for example "mean radius")
    label    = the answer we want to predict (1 = malignant, 0 = benign)
    model    = a math recipe that learns a pattern from examples
    training = showing the model examples so it can learn
    test set = examples we hide during training and use only at the end

The "# %%" lines split the file into cells. Editors like VS Code let you run
one cell at a time, and the notebook in notebooks/ is built from these cells.
"""

# %% [markdown]
# ## 1. Import the tools (libraries) we need
# A "library" is code someone else wrote that we can reuse.
# - numpy / pandas: work with numbers and tables
# - matplotlib / seaborn: draw charts
# - scikit-learn (sklearn): the main machine learning toolbox
# - xgboost: a strong model type called "gradient boosted trees"
# - joblib: saves a trained model to a file

# %%
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Folders where outputs go. Path(__file__) is this file; .parent.parent is the project folder.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # inside a notebook there is no __file__, so use the folder above notebooks/
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
IMAGES = ROOT / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
for folder in (IMAGES, RESULTS, MODELS):
    folder.mkdir(exist_ok=True)

# A "random seed" makes anything random (like shuffling) happen the same way every run,
# so you and anyone checking your work get the same numbers.
SEED = 42

# %% [markdown]
# ## 2. Load the data
# The dataset is the UCI Breast Cancer Wisconsin (Diagnostic) dataset.
# It ships inside scikit-learn, so there is nothing to download.
# 569 patients (rows), 30 measurements each (columns).

# %%
data = load_breast_cancer(as_frame=True)
X = data.data  # the 30 feature columns, as a pandas table (a "DataFrame")

# In this dataset, target 0 means malignant and 1 means benign.
# We flip it so that 1 = malignant, because the cancer case is the one we care
# about catching. "Positive" in the scores below will then mean "malignant".
y = 1 - data.target
y.name = "malignant"

print("Rows (patients), columns (features):", X.shape)
print("Malignant (1) vs benign (0) counts:")
print(y.value_counts())
print("\nFirst 5 rows:")
print(X.head())

# %% [markdown]
# ## 3. Explore the data (EDA = Exploratory Data Analysis)
# Before training, look at the data. Two charts:
# 1. How many of each class we have (class balance).
# 2. Which features move together (correlation). Values near +1 or -1 mean two
#    features carry nearly the same information.

# %%
print("Missing values in the whole table:", int(X.isna().sum().sum()))
print(X.describe().T[["mean", "std", "min", "max"]].head(10))

fig, ax = plt.subplots(figsize=(5, 4))
y.map({0: "Benign", 1: "Malignant"}).value_counts().plot(kind="bar", ax=ax, color=["#4C9F70", "#C8553D"])
ax.set_title("Class balance")
ax.set_ylabel("Number of patients")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(IMAGES / "class_balance.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("How the 30 features correlate with each other")
plt.tight_layout()
plt.savefig(IMAGES / "correlation_heatmap.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Split into training and test sets
# We hide 20% of patients as a test set. The models never see them while learning.
# stratify=y keeps the same malignant/benign ratio in both parts.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("Training patients:", len(X_train), "| Test patients:", len(X_test))

# %% [markdown]
# ## 5. Build the four models
# A Pipeline chains steps. For Logistic Regression and SVM we first SCALE the
# features (so a column measured in thousands doesn't drown out one measured in
# fractions), then fit the model. Tree models (Random Forest, XGBoost) don't
# need scaling, because they only ask "is this value above or below a cutoff?".
#
# - Logistic Regression: draws a straight boundary between the two classes.
# - SVM (Support Vector Machine): finds the boundary with the widest safety gap.
# - Random Forest: hundreds of decision trees vote; the majority wins.
# - XGBoost: trees built one after another, each fixing the last one's mistakes.

# %%
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
    ]),
    "SVM": Pipeline([
        ("scale", StandardScaler()),
        # An SVM gives a yes/no answer but not a probability. CalibratedClassifierCV
        # wraps it so it can also say "83% chance malignant", which ROC curves need.
        ("model", CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False)),
    ]),
    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=300, random_state=SEED)),
    ]),
    "XGBoost": Pipeline([
        ("model", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                eval_metric="logloss", random_state=SEED)),
    ]),
}

# %% [markdown]
# ## 6. Cross-validation on the training set only
# Cross-validation (CV) splits the training data into 5 parts, trains on 4,
# checks on the 5th, and rotates 5 times. The average is a fairer score than a
# single split, and we still haven't touched the test set.
# ROC-AUC: 1.0 is perfect ranking of sick vs healthy, 0.5 is a coin flip.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"{name:20s} CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

# %% [markdown]
# ## 7. Tune the settings (hyperparameters) of two models
# Hyperparameters are knobs we choose before training, like how strict the
# model is. GridSearchCV tries every combination in a small grid with CV and
# keeps the best. We tune Logistic Regression (knob C) and SVM (C and gamma).

# %%
grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    # "model__estimator__C" means: inside the step named "model", the wrapped SVM's C knob.
    "SVM": {"model__estimator__C": [0.1, 1, 10, 100], "model__estimator__gamma": ["scale", 0.01, 0.001]},
}
for name, grid in grids.items():
    search = GridSearchCV(models[name], grid, cv=cv, scoring="roc_auc")
    search.fit(X_train, y_train)
    print(f"{name}: best settings {search.best_params_}, CV ROC-AUC {search.best_score_:.4f}")
    models[name] = search.best_estimator_  # keep the tuned version

# %% [markdown]
# ## 8. Final test: train on all training data, score on the hidden test set
# Metrics (with "positive" = malignant):
# - Accuracy: share of all predictions that are right.
# - Precision: of patients we flagged malignant, the share that truly were.
# - Recall: of truly malignant patients, the share we caught. Most important here,
#   because missing a cancer is the worst mistake.
# - F1: one number balancing precision and recall.
# - ROC-AUC: how well the model ranks malignant above benign.

# %%
rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)  # 0 or 1 for each test patient
    prob = pipe.predict_proba(X_test)[:, 1]  # probability of malignant
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    })

results = pd.DataFrame(rows).sort_values(["recall", "roc_auc"], ascending=False).round(4)
results.to_csv(RESULTS / "results.csv", index=False)
print(results.to_string(index=False))

# The "best" model: highest recall (fewest missed cancers), ties broken by ROC-AUC.
best_name = results.iloc[0]["model"]
best_model = models[best_name]
print("\nBest model:", best_name)
print(classification_report(y_test, best_model.predict(X_test), target_names=["Benign", "Malignant"]))

# %% [markdown]
# ## 9. Charts for the results
# - Confusion matrix: rows are the truth, columns the prediction. The off-diagonal
#   boxes are mistakes. The bottom-left box (malignant predicted as benign) is the
#   dangerous one.
# - ROC curves: all four models on one chart. Closer to the top-left is better.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                      display_labels=["Benign", "Malignant"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig(IMAGES / "confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in models.items():
    prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)  # false/true positive rate at every threshold
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Coin flip")
ax.set_xlabel("False positive rate (benign wrongly flagged)")
ax.set_ylabel("True positive rate (malignant caught)")
ax.set_title("ROC curves on the test set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(IMAGES / "roc_curves.png", dpi=150)
plt.close()

# %% [markdown]
# ## 10. Which features matter most?
# A Random Forest can report how much each feature helped its decisions
# ("feature importance"). This helps explain the model in your video.

# %%
rf = models["Random Forest"].named_steps["model"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("Top 10 features:\n", importance.round(4))

fig, ax = plt.subplots(figsize=(7, 5))
importance.sort_values().plot(kind="barh", ax=ax, color="#3A6EA5")
ax.set_title("Top 10 features (Random Forest importance)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(IMAGES / "feature_importance.png", dpi=150)
plt.close()

# %% [markdown]
# ## 11. Save the best model
# joblib writes the trained model to a file, so predict.py can use it later
# without training again. We also save the column names it expects.

# %%
joblib.dump({"model": best_model, "name": best_name, "features": list(X.columns)},
            MODELS / "best_model.joblib")
print("Saved", MODELS / "best_model.joblib")
