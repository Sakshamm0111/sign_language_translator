import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import time
from collections import deque, Counter

# ── Normalization (matches collect_data.py exactly) ───────────
def normalize_landmarks(raw):
    lm = raw.reshape(21, 3)
    lm = lm - lm[0]                        # wrist at origin
    hand_size = np.max(np.abs(lm)) + 1e-6
    return (lm / hand_size).flatten()       # (63,)

# ── TTS — runs in background thread, never blocks ─────────────
SPEAK_MAP = {
    "Fck u": "fuck you",
}

def speak(text):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()

def _speak(text):
    eng = pyttsx3.init()
    eng.setProperty('rate', 150)
    eng.say(text)
    eng.runAndWait()

# ── Load model ────────────────────────────────────────────────
with open("models/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

n_features = model.n_features_in_
classes    = list(model.classes_)
print(f"✅ Model loaded. Features: {n_features}  |  Classes: {len(classes)}")

# ── MediaPipe ─────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Config ────────────────────────────────────────────────────
CONFIDENCE_TH  = 0.60   # per-frame confidence threshold
HOLD_TIME      = 1.0    # seconds to hold sign before adding
BUFFER_SIZE    = 12     # rolling window — number of frames to vote over
VOTE_TH        = 0.70   # fraction of buffer that must agree on same sign

# ── State ─────────────────────────────────────────────────────
sentence          = []
last_sign         = None
sign_start        = time.time()
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

print("✅ Camera opened. Show your hand!")
print("   Q=quit  C=clear sentence  S=speak sentence  SPACE=undo last word")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    sign       = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        raw = np.array([[lm.x, lm.y, lm.z]
                        for lm in hand_lm.landmark]).flatten()  # (63,)

        if raw.shape[0] != n_features:
            cv2.putText(frame, f"⚠ Feature mismatch: {raw.shape[0]} vs {n_features}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            proba      = model.predict_proba([normalize_landmarks(raw)])[0]
            confidence = float(proba.max())
            sign       = classes[int(proba.argmax())]

            # ── Feed rolling buffer ────────────────────────────
            if confidence >= CONFIDENCE_TH:
                prediction_buffer.append(sign)
            else:
                prediction_buffer.clear()   # low confidence → hard reset

            # ── Stable vote from buffer ────────────────────────
            stable_sign = None
            if len(prediction_buffer) == BUFFER_SIZE:
                top, count = Counter(prediction_buffer).most_common(1)[0]
                if count / BUFFER_SIZE >= VOTE_TH:
                    stable_sign = top

            # Top-2 debug info
            top2_idx  = proba.argsort()[-2:][::-1]
            top2_info = "  ".join([f"{classes[i]}:{proba[i]:.0%}" for i in top2_idx])

            # Color by confidence
            if confidence >= CONFIDENCE_TH:
                color = (0, 220, 0)
            elif confidence >= 0.40:
                color = (0, 165, 255)
            else:
                color = (0, 0, 220)

            # Confidence bar
            bar_w = int(confidence * 250)
            cv2.rectangle(frame, (10, 88), (10 + bar_w, 108), color, -1)
            cv2.rectangle(frame, (10, 88), (260, 108), (255, 255, 255), 1)

            # Sign label (raw per-frame prediction)
            cv2.putText(frame, f"{sign}  ({confidence:.0%})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

            # Stable indicator — shows what the buffer has locked onto
            if stable_sign:
                cv2.putText(frame, f"[{stable_sign}]",
                            (frame.shape[1] - 140, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

            # Top-2 hint
            cv2.putText(frame, top2_info,
                        (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ── Hold-to-confirm (uses stable_sign, not raw sign) ──
            active = stable_sign if stable_sign else (sign if confidence >= CONFIDENCE_TH else None)

            if active:
                now = time.time()
                if active != last_sign:
                    last_sign  = active
                    sign_start = now
                else:
                    held   = now - sign_start
                    prog_w = int(min(held / HOLD_TIME, 1.0) * 250)
                    cv2.rectangle(frame, (10, 110), (10 + prog_w, 118), (255, 255, 0), -1)

                    if held >= HOLD_TIME:
                        if not sentence or sentence[-1] != active:
                            sentence.append(active)
                            print(f"✅ Added: '{active}'  →  {' '.join(sentence)}")
                            speak(SPEAK_MAP.get(active, active))
                        sign_start = now
            else:
                if sign != last_sign:
                    last_sign  = sign
                    sign_start = time.time()

    else:
        # No hand → clear buffer
        prediction_buffer.clear()

    # Sentence bar
    display = " ".join(sentence[-8:]) if sentence else "(no signs yet)"
    cv2.rectangle(frame, (0, frame.shape[0] - 55),
                  (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(frame, display,
                (10, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Controls hint
    cv2.putText(frame, "Q=quit  C=clear  S=speak  SPACE=undo",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence.clear()
        print("🗑 Sentence cleared")
    elif key == ord('s') and sentence:
        full = " ".join(SPEAK_MAP.get(w, w) for w in sentence)
        print(f"🔊 Speaking: '{full}'")
        speak(full)
    elif key == ord(' ') and sentence:
        removed = sentence.pop()
        print(f"↩ Removed: '{removed}'  →  {' '.join(sentence)}")

cap.release()
cv2.destroyAllWindows()
print("👋 Translator closed.")
print(f"   Final sentence: {' '.join(sentence)}")