import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────
def load_data():
    print("[INFO] Loading data...")
    data, labels, signs = [], [], []

    for sign in sorted(os.listdir(DATA_DIR)):
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue

        files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        if len(files) == 0:
            print(f"[WARN] No .npy files found in {sign_dir} — skipping")
            continue

        signs.append(sign)
        print(f"[INFO] Loading '{sign}' — {len(files)} samples")

        for file in files:
            file_path = os.path.join(sign_dir, file)
            try:
                landmarks = np.load(file_path)
                if landmarks.shape[0] != 63:
                    print(f"[WARN] Skipping {file_path} — shape {landmarks.shape} (expected 63)")
                    continue
                data.append(landmarks)
                labels.append(sign)
            except Exception as e:
                print(f"[WARN] Could not load {file_path}: {e}")

    print(f"\n[INFO] Total samples: {len(data)}")
    print(f"[INFO] Total signs  : {len(signs)}")
    return np.array(data), np.array(labels), signs

# ── Train ──────────────────────────────────────────────────────
def train_model():
    X, y, signs = load_data()

    if len(X) == 0:
        print("[ERROR] No data! Run collect_data.py first.")
        return

    # Verify feature count
    print(f"\n[INFO] Feature shape: {X.shape}  (expected: N × 63)")
    assert X.shape[1] == 63, f"Wrong feature count: {X.shape[1]}"

    print("\n[INFO] Splitting 80/20 train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Pipeline: scaler is BAKED INTO the model ───────────────
    # This means translator.py never needs to normalize anything.
    # Feed raw MediaPipe landmarks → Pipeline handles the rest.
    print("\n[INFO] Building Pipeline: StandardScaler + RandomForest...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,    # More trees = more stable
            max_depth=None,      # Let trees grow fully
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("[INFO] Training...")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete!")

    # ── Accuracy ───────────────────────────────────────────────
    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {accuracy * 100:.2f}%")

    # Cross-validation for more robust estimate
    print("[INFO] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"[RESULT] CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    print("\n[REPORT] Per-sign accuracy:")
    print(classification_report(y_test, y_pred))

    # ── Save pipeline ──────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "sign_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[SAVED] Pipeline saved → {model_path}")

    # Verify it loads and predicts correctly
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    test_sample = X_test[0:1]
    pred = loaded.predict(test_sample)
    prob = loaded.predict_proba(test_sample).max()
    print(f"[VERIFY] Load test — predicted: {pred[0]}  confidence: {prob:.2%}  ✅")

    # ── Accuracy chart ─────────────────────────────────────────
    print("\n[INFO] Generating per-sign accuracy chart...")
    sign_accuracy = {}
    for sign in signs:
        mask = y_test == sign
        if mask.sum() > 0:
            sign_accuracy[sign] = accuracy_score(y_test[mask], y_pred[mask]) * 100

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['#22c55e' if v >= 90 else '#f59e0b' if v >= 80 else '#ef4444'
              for v in sign_accuracy.values()]
    ax.bar(sign_accuracy.keys(), sign_accuracy.values(), color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=90, color='#22c55e', linestyle='--', alpha=0.7, label='90% (great)')
    ax.axhline(y=80, color='#f59e0b', linestyle='--', alpha=0.7, label='80% (ok)')
    ax.set_title(f'Per-Sign Accuracy  |  Overall: {accuracy * 100:.1f}%', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sign')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(MODEL_DIR, "accuracy_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"[SAVED] Accuracy chart → {chart_path}")
    plt.show()

    return accuracy

# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("=" * 50)
    accuracy = train_model()

    if accuracy:
        print("\n" + "=" * 50)
        print(f"  Final accuracy: {accuracy * 100:.2f}%")
        if accuracy >= 0.95:
            print("  🟢 EXCELLENT! Run translator.py")
        elif accuracy >= 0.80:
            print("  🟡 GOOD. Collect more data per sign for higher accuracy.")
        else:
            print("  🔴 LOW. Re-collect data — check lighting/hand position.")
        print("=" * 50)

    input("\nPress Enter to exit...")