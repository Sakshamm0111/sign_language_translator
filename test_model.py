"""
test_model.py — Diagnostic tool for the sign language Pipeline model.
"""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/sign_model.pkl"
DATA_DIR   = "data"

print("=" * 55)
print("  SIGN LANGUAGE MODEL DIAGNOSTIC")
print("=" * 55)

# ── 1. Load model ──────────────────────────────────────────────
print("\n[1/5] Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

model_type = type(model).__name__
print(f"  Model type        : {model_type}")
print(f"  Expected features : {model.n_features_in_}")
print(f"  Classes ({len(model.classes_)})     : {list(model.classes_)}")

is_pipeline = model_type == "Pipeline"
print(f"  Is Pipeline       : {'✅ Yes — scaler baked in' if is_pipeline else '⚠ No — plain classifier'}")

# ── 2. Check .npy samples ──────────────────────────────────────
print("\n[2/5] Checking saved .npy samples...")
signs_to_test = ['A', 'B', 'hello', 'yes']  # spot-check these
all_good = True

for label in signs_to_test:
    sample_path = os.path.join(DATA_DIR, label, "0.npy")
    if not os.path.exists(sample_path):
        print(f"  ⚠ '{label}' sample not found at {sample_path}")
        all_good = False
        continue

    raw = np.load(sample_path)
    vmin, vmax = raw.min(), raw.max()
    in_range = 0.0 <= vmin and vmax <= 1.0

    print(f"  '{label}': shape={raw.shape}  range=[{vmin:.4f}, {vmax:.4f}]  "
          f"{'✅ raw MediaPipe coords' if in_range else '⚠ unexpected range'}")

# ── 3. Predict on saved samples ────────────────────────────────
print("\n[3/5] Prediction test on saved samples...")
signs_available = sorted([d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d))])
correct = 0
total   = 0

for label in signs_available:
    sample_path = os.path.join(DATA_DIR, label, "0.npy")
    if not os.path.exists(sample_path):
        continue
    raw  = np.load(sample_path)
    pred = model.predict([raw])[0]
    prob = model.predict_proba([raw]).max()
    ok   = pred == label
    if ok:
        correct += 1
    total += 1
    status = "✅" if ok else f"❌ (predicted '{pred}')"
    print(f"  '{label}': {prob:.0%}  {status}")

if total > 0:
    print(f"\n  Sample accuracy: {correct}/{total} = {correct/total:.0%}")

# ── 4. Confusion matrix on held-out test set ───────────────────
print("\n[4/5] Building confusion matrix on held-out test set...")

# Load ALL real samples (no aug_ files — we want true generalization)
X, y = [], []
for sign in signs_available:
    sign_dir = os.path.join(DATA_DIR, sign)
    real_files = [f for f in os.listdir(sign_dir)
                  if f.endswith('.npy') and not f.startswith('aug_')]
    for fname in real_files:
        raw = np.load(os.path.join(sign_dir, fname))
        if raw.shape[0] == 63:
            X.append(raw)
            y.append(sign)

X = np.array(X)
y = np.array(y)

if len(X) > 0:
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    print(f"\n  Test set size : {len(X_test)} samples")
    print(f"  Test accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\n  Per-sign breakdown:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Confusion matrix plot ──────────────────────────────────
    labels = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(18, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title("Confusion Matrix — Sign Language Model\n"
                 "(off-diagonal = misclassifications to investigate)",
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    cm_path = "models/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\n  ✅ Confusion matrix saved → {cm_path}")

    # ── Print top confusable pairs ─────────────────────────────
    print("\n  Top confusable sign pairs (check these first):")
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], true_label, pred_label))
    pairs.sort(reverse=True)

    for count, true_l, pred_l in pairs[:10]:
        print(f"    '{true_l}' → predicted as '{pred_l}' : {count} times")

    plt.show(block=False)
    plt.pause(0.1)
else:
    print("  [WARN] Not enough data to build confusion matrix.")

# ── 5. Verdict ─────────────────────────────────────────────────
print("\n[5/5] Verdict:")
if correct == total and total > 0:
    print("  ✅ All spot-checks passed — model looks great!")
    print("  ✅ Ready to run: python translator.py")
elif total > 0 and correct / total >= 0.8:
    print("  🟡 Most checks passed. Some signs may underperform in real time.")
    print("     Check the confusion matrix to see which pairs to re-collect.")
else:
    print("  ❌ Too many failures — likely a data/normalization mismatch.")
    print("     Steps to fix:")
    print("     1. Delete the 'data/' folder entirely")
    print("     2. Run: python collect_data.py")
    print("     3. Run: python train_model.py")
    print("     4. Run: python test_model.py  ← should pass now")

print("\n" + "=" * 55)
input("Press Enter to exit...")
