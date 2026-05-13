import cv2
import mediapipe as mp
import numpy as np
import os
 
DATA_DIR = "data"
SAMPLES_PER_SIGN = 150  # Increased from 100 → better generalization
 
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
 
# ── Signs list ────────────────────────────────────────────────
SIGNS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'hello','thanks','yes','no','please','sorry','help','good','bad','love'
]
 
# NOTE: Removed 'Fck u' — keep dataset professional and consistent
 
cap = cv2.VideoCapture(0)
print("=" * 50)
print("  SIGN LANGUAGE DATA COLLECTION")
print("=" * 50)
print("Controls: SPACE = start collecting this sign")
print("          Q     = quit early")
print(f"Saving RAW landmarks (NO normalization)")
print(f"Samples per sign: {SAMPLES_PER_SIGN}")
print(f"Total signs: {len(SIGNS)}")
print(f"Total samples to collect: {len(SIGNS) * SAMPLES_PER_SIGN}")
print("=" * 50)
 
collected_signs = 0
 
for sign in SIGNS:
    sign_dir = os.path.join(DATA_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)
 
    print(f"\n[{collected_signs + 1}/{len(SIGNS)}] GET READY: '{sign.upper()}'")
    print(f"    Position your hand and press SPACE to start")
 
    # ── Wait for SPACE ─────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands.process(rgb)
 
        # Show live hand landmarks while waiting
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Hand detected!", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
        cv2.putText(frame, f"SIGN: {sign.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "Press SPACE to collect", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Progress: {collected_signs}/{len(SIGNS)} signs done",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
 
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            print("\n[QUIT] Exiting early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
 
    # ── Collect samples ────────────────────────────────────────
    count = 0
    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
 
        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
 
            # ── Save RAW landmarks — NO normalization ──────────
            raw = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand_lm.landmark]).flatten()  # (63,)
 
            file_path = os.path.join(sign_dir, f"{count}.npy")
            np.save(file_path, raw)
            count += 1
 
        # Progress bar
        progress = count / SAMPLES_PER_SIGN
        bar_w    = int(progress * 300)
        cv2.rectangle(frame, (10, 115), (10 + bar_w, 132), (0, 220, 0), -1)
        cv2.rectangle(frame, (10, 115), (310, 132), (255, 255, 255), 1)
 
        cv2.putText(frame, f"Collecting: {sign.upper()} [{count}/{SAMPLES_PER_SIGN}]",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"{int(progress * 100)}%",
                    (320, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)
 
    print(f"    ✅ '{sign}' — {count} samples saved")
    collected_signs += 1
 
cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 50)
print(f"✅ ALL DONE! {collected_signs} signs × {SAMPLES_PER_SIGN} = {collected_signs * SAMPLES_PER_SIGN} samples")
print("   Next step: python train_model.py")
print("=" * 50)
  