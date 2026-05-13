"""
augment_data.py — Synthetically expand training data without re-recording.

How it works:
  - Loads every real .npy sample from data/<sign>/
  - Generates AUGMENTS_PER variations per sample using:
      noise   → simulates hand tremor / sensor jitter
      scale   → simulates hand closer/further from camera
      shift   → simulates wrist not perfectly centered
  - Saves augmented samples as aug_0.npy, aug_1.npy, ... alongside originals

Result: 150 real samples → ~750 total per sign (150 + 150×5 augmented)

Run order:
  1. python augment_data.py   ← adds aug_*.npy files to data/
  2. python train_model.py    ← loads everything automatically
  3. python test_model.py     ← verify accuracy improved
  4. python translator.py     ← run the translator
"""

import numpy as np
import os

DATA_DIR     = "data"
AUGMENTS_PER = 5              # augmented copies per real sample
NOISE_STD    = 0.01           # gaussian noise std  (hand jitter)
SCALE_RANGE  = (0.85, 1.15)  # uniform scale range (distance variation)
SHIFT_RANGE  = (-0.05, 0.05) # uniform shift range (wrist centering variation)

np.random.seed(42)


def augment_sample(raw):
    """
    Generate AUGMENTS_PER variations of one raw (63,) landmark array.
    raw = [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20] — MediaPipe coords
    """
    results = []
    for _ in range(AUGMENTS_PER):
        aug = raw.copy().astype(np.float64)

        # 1. Gaussian noise — simulates sensor jitter / slight tremor
        aug += np.random.normal(0, NOISE_STD, aug.shape)

        # 2. Uniform scale — simulates hand closer or further from camera
        aug *= np.random.uniform(*SCALE_RANGE)

        # 3. x/y shift — simulates wrist not perfectly at MediaPipe origin
        #    x = indices 0,3,6,...  y = indices 1,4,7,...  z left alone
        shift_x = np.random.uniform(*SHIFT_RANGE)
        shift_y = np.random.uniform(*SHIFT_RANGE)
        aug[0::3] += shift_x   # every x coord
        aug[1::3] += shift_y   # every y coord

        results.append(aug.astype(np.float32))
    return results


def augment_sign(sign_dir, sign_name):
    """Augment all real samples in one sign folder."""
    real_files = [f for f in os.listdir(sign_dir)
                  if f.endswith('.npy') and not f.startswith('aug_')]

    if not real_files:
        print(f"  [WARN] No real samples found in {sign_dir} — skipping")
        return 0, 0

    # Remove old augmented files so we don't double-augment on re-runs
    old_aug = [f for f in os.listdir(sign_dir) if f.startswith('aug_')]
    for f in old_aug:
        os.remove(os.path.join(sign_dir, f))

    aug_count = 0
    for fname in real_files:
        raw = np.load(os.path.join(sign_dir, fname))
        if raw.shape[0] != 63:
            print(f"  [WARN] Skipping {fname} — shape {raw.shape} (expected 63,)")
            continue

        variations = augment_sample(raw)
        for i, var in enumerate(variations):
            stem     = fname.replace('.npy', '')
            aug_path = os.path.join(sign_dir, f"aug_{stem}_{i}.npy")
            np.save(aug_path, var)
            aug_count += 1

    return len(real_files), aug_count


def main():
    print("=" * 55)
    print("  SIGN LANGUAGE DATA AUGMENTATION")
    print("=" * 55)
    print(f"  Augments per sample : {AUGMENTS_PER}")
    print(f"  Noise std           : {NOISE_STD}")
    print(f"  Scale range         : {SCALE_RANGE}")
    print(f"  Shift range         : {SHIFT_RANGE}")
    print("=" * 55)

    signs = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d))])

    if not signs:
        print(f"\n[ERROR] No sign folders found in '{DATA_DIR}/'")
        print("        Run collect_data.py first.")
        return

    total_real = 0
    total_aug  = 0

    for sign in signs:
        sign_dir = os.path.join(DATA_DIR, sign)
        real, aug = augment_sign(sign_dir, sign)
        total_real += real
        total_aug  += aug
        print(f"  '{sign}': {real} real → +{aug} augmented  "
              f"(total: {real + aug})")

    print("\n" + "=" * 55)
    print(f"  Real samples      : {total_real}")
    print(f"  Augmented added   : {total_aug}")
    print(f"  Grand total       : {total_real + total_aug}")
    print("=" * 55)
    print("\n✅ Done! Now retrain:")
    print("     python train_model.py")
    print("   Then check accuracy:")
    print("     python test_model.py")


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")