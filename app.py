"""
app.py — Flask Web Dashboard for Sign Language Translator
Run: python app.py
Open: http://localhost:5000
"""
from flask import Flask, Response, render_template_string, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
import os
import signal
from collections import deque, Counter

app = Flask(__name__)

# ── Normalization (must match collect_data.py exactly) ────────
def normalize_landmarks(raw):
    lm = raw.reshape(21, 3)
    lm = lm - lm[0]                        # wrist at origin
    hand_size = np.max(np.abs(lm)) + 1e-6
    return (lm / hand_size).flatten()       # (63,)

# ── Load model ────────────────────────────────────────────────
with open("models/sign_model.pkl", "rb") as f:
    model = pickle.load(f)
print(f"✅ Model loaded. Classes: {list(model.classes_)}")

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
CONFIDENCE_TH = 0.60   # per-frame threshold
HOLD_TIME     = 1.5    # seconds to hold before adding to sentence
BUFFER_SIZE   = 12     # rolling window frames
VOTE_TH       = 0.70   # fraction of buffer that must agree

# ── Shared state (thread-safe) ────────────────────────────────
state_lock        = threading.Lock()
prediction_buffer = deque(maxlen=BUFFER_SIZE)

state = {
    "current_sign": "",
    "confidence": 0.0,
    "sentence": [],
    "last_added": "",
    "hand_detected": False,
    "top2": []
}

last_sign  = None
sign_start = time.time()

# ── Video processing ──────────────────────────────────────────
cap = cv2.VideoCapture(0)

def process_frame(frame):
    global last_sign, sign_start

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    sign          = ""
    confidence    = 0.0
    hand_detected = False
    top2          = []
    stable_sign   = None

    if results.multi_hand_landmarks:
        hand_detected = True
        hand_lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        raw = np.array([[lm.x, lm.y, lm.z]
                        for lm in hand_lm.landmark]).flatten()  # (63,)

        if raw.shape[0] == model.n_features_in_:
            normed     = normalize_landmarks(raw)
            proba      = model.predict_proba([normed])[0]
            confidence = float(proba.max())
            sign       = model.classes_[int(proba.argmax())]
            top2_idx   = proba.argsort()[-2:][::-1]
            top2       = [(model.classes_[i], float(proba[i])) for i in top2_idx]

            # ── Rolling window ─────────────────────────────
            if confidence >= CONFIDENCE_TH:
                prediction_buffer.append(sign)
            else:
                prediction_buffer.clear()

            if len(prediction_buffer) == BUFFER_SIZE:
                top_sign, count = Counter(prediction_buffer).most_common(1)[0]
                if count / BUFFER_SIZE >= VOTE_TH:
                    stable_sign = top_sign

            # Color by confidence
            if confidence >= CONFIDENCE_TH:
                color = (0, 220, 0)
            elif confidence >= 0.40:
                color = (0, 165, 255)
            else:
                color = (0, 80, 220)

            # Overlay
            cv2.putText(frame, f"{sign}  {confidence:.0%}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            bar_w = int(confidence * 200)
            cv2.rectangle(frame, (10, 60), (10 + bar_w, 76), color, -1)
            cv2.rectangle(frame, (10, 60), (210, 76), (255, 255, 255), 1)

            # Stable sign indicator
            if stable_sign:
                cv2.putText(frame, f"[{stable_sign}]",
                            (frame.shape[1] - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

            # ── Hold-to-confirm (prefers stable over raw) ──
            active = stable_sign if stable_sign else (sign if confidence >= CONFIDENCE_TH else None)

            if active:
                now = time.time()
                if active != last_sign:
                    last_sign  = active
                    sign_start = now
                else:
                    held   = now - sign_start
                    prog_w = int(min(held / HOLD_TIME, 1.0) * 200)
                    cv2.rectangle(frame, (10, 78), (10 + prog_w, 88), (255, 255, 0), -1)
                    if held >= HOLD_TIME:
                        with state_lock:
                            if not state["sentence"] or state["sentence"][-1] != active:
                                state["sentence"].append(active)
                                state["last_added"] = active
                        sign_start = now
            else:
                if sign != last_sign:
                    last_sign  = sign
                    sign_start = time.time()

    else:
        prediction_buffer.clear()

    # Update shared state
    with state_lock:
        state["current_sign"]  = sign
        state["confidence"]    = confidence
        state["hand_detected"] = hand_detected
        state["top2"]          = top2

    # Sentence overlay on frame
    with state_lock:
        sentence_text = " ".join(state["sentence"][-6:])
    cv2.rectangle(frame, (0, frame.shape[0] - 45),
                  (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
    cv2.putText(frame, sentence_text or "— no signs yet —",
                (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(dict(state))

@app.route('/clear', methods=['POST'])
def clear():
    with state_lock:
        state["sentence"].clear()
        state["last_added"] = ""
    return jsonify({"ok": True})

@app.route('/undo', methods=['POST'])
def undo():
    with state_lock:
        if state["sentence"]:
            state["sentence"].pop()
    return jsonify({"ok": True})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cap.release()   # release camera before shutting down
    def _kill():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"ok": True})


# ── HTML Template ─────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign Language Translator</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06d6a0;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --green: #22c55e;
    --red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface);
  }
  header .logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent2);
    letter-spacing: -0.5px;
  }
  header .subtitle {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 2px;
  }
  header .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    margin-left: auto;
    transition: background 0.3s;
  }
  header .status-dot.active { background: var(--green); box-shadow: 0 0 8px var(--green); }
  header .shutdown-btn {
    margin-left: 16px;
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #3f1515;
    background: transparent;
    color: var(--red);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  header .shutdown-btn:hover { background: #3f1515; }

  main {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 0;
    flex: 1;
    overflow: hidden;
  }

  .video-panel {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .video-panel img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }

  .side-panel {
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .panel-section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }
  .panel-section h3 {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .sign-display { text-align: center; }
  .sign-big {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: var(--accent2);
    line-height: 1;
    min-height: 72px;
    transition: all 0.2s;
    letter-spacing: -2px;
  }
  .sign-big.no-hand { color: var(--border); font-size: 2rem; }

  .confidence-bar-wrap {
    margin-top: 12px;
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
  }
  .confidence-bar {
    height: 100%;
    border-radius: 4px;
    background: var(--muted);
    transition: width 0.15s, background 0.2s;
  }
  .confidence-bar.high { background: var(--green); }
  .confidence-bar.med  { background: var(--warn); }
  .confidence-bar.low  { background: var(--red); }

  .conf-label {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 0.75rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  .top2 { display: flex; gap: 8px; margin-top: 12px; }
  .top2-chip {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .top2-chip.best { border-color: var(--accent); }
  .top2-chip .chip-sign {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: var(--text);
  }
  .top2-chip .chip-prob { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }

  .sentence-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    min-height: 80px;
    padding: 14px;
    font-size: 1.1rem;
    line-height: 1.6;
    word-break: break-word;
  }
  .sentence-box .word { display: inline-block; }
  .sentence-box .word.new {
    color: var(--accent2);
    animation: popIn 0.4s ease;
  }
  @keyframes popIn {
    0%   { transform: scale(0.7); opacity: 0.3; }
    60%  { transform: scale(1.15); }
    100% { transform: scale(1); opacity: 1; }
  }

  .btn-row { display: flex; gap: 8px; margin-top: 12px; }
  .btn {
    flex: 1;
    padding: 9px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 600;
  }
  .btn:hover { background: var(--border); }
  .btn.danger:hover { background: #3f1515; border-color: var(--red); color: var(--red); }
  .btn.accent { background: var(--accent); border-color: var(--accent); }
  .btn.accent:hover { opacity: 0.85; }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 180px;
    overflow-y: auto;
  }
  .history-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.82rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .history-item .ts {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--border);
  }
  .empty-state {
    color: var(--muted);
    font-size: 0.8rem;
    font-style: italic;
    text-align: center;
    padding: 20px 0;
  }

  /* Shutdown overlay */
  #shutdownOverlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    z-index: 999;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
  }
  #shutdownOverlay.show { display: flex; }
  #shutdownOverlay p {
    font-family: 'Space Mono', monospace;
    color: var(--red);
    font-size: 1.1rem;
  }
  #shutdownOverlay small { color: var(--muted); font-size: 0.8rem; }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">✋ Sign Translator</div>
    <div class="subtitle">Real-time ASL Recognition</div>
  </div>
  <div class="status-dot" id="statusDot"></div>
  <button class="shutdown-btn" onclick="shutdown()">⏹ Stop Server</button>
</header>

<main>
  <div class="video-panel">
    <img src="/video_feed" alt="Live Camera Feed">
  </div>

  <div class="side-panel">

    <div class="panel-section">
      <h3>Current Sign</h3>
      <div class="sign-display">
        <div class="sign-big no-hand" id="signBig">—</div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar" id="confBar" style="width:0%"></div>
        </div>
        <div class="conf-label">
          <span>confidence</span>
          <span id="confPct">0%</span>
        </div>
        <div class="top2" id="top2"></div>
      </div>
    </div>

    <div class="panel-section">
      <h3>Sentence</h3>
      <div class="sentence-box" id="sentenceBox">
        <span class="empty-state">Hold a sign for 1.5s to add it...</span>
      </div>
      <div class="btn-row">
        <button class="btn danger" onclick="clearSentence()">🗑 Clear</button>
        <button class="btn" onclick="undoLast()">↩ Undo</button>
        <button class="btn accent" onclick="speakSentence()">🔊 Speak</button>
      </div>
    </div>

    <div class="panel-section">
      <h3>Session History</h3>
      <div class="history-list" id="historyList">
        <div class="empty-state">Completed sentences appear here</div>
      </div>
    </div>

  </div>
</main>

<!-- Shutdown overlay -->
<div id="shutdownOverlay">
  <p>⏹ Server stopped</p>
  <small>Camera released. You can close this tab.</small>
</div>

<script>
  let prevSentence  = [];
  let prevLastAdded = "";
  const history     = [];

  function updateUI(s) {
    document.getElementById('statusDot').className =
      'status-dot' + (s.hand_detected ? ' active' : '');

    const signEl = document.getElementById('signBig');
    if (s.current_sign && s.confidence >= 0.50) {
      signEl.textContent = s.current_sign.toUpperCase();
      signEl.className   = 'sign-big';
    } else {
      signEl.textContent = s.hand_detected ? '...' : '—';
      signEl.className   = 'sign-big no-hand';
    }

    const bar = document.getElementById('confBar');
    const pct = document.getElementById('confPct');
    const c   = s.confidence;
    bar.style.width = (c * 100) + '%';
    bar.className   = 'confidence-bar ' + (c >= 0.8 ? 'high' : c >= 0.5 ? 'med' : 'low');
    pct.textContent = Math.round(c * 100) + '%';

    const top2El = document.getElementById('top2');
    if (s.top2 && s.top2.length) {
      top2El.innerHTML = s.top2.map((item, i) => `
        <div class="top2-chip ${i === 0 ? 'best' : ''}">
          <div class="chip-sign">${item[0].toUpperCase()}</div>
          <div class="chip-prob">${Math.round(item[1] * 100)}%</div>
        </div>
      `).join('');
    } else {
      top2El.innerHTML = '';
    }

    const box = document.getElementById('sentenceBox');
    if (!s.sentence || s.sentence.length === 0) {
      box.innerHTML = '<span class="empty-state">Hold a sign for 1.5s to add it...</span>';
    } else {
      box.innerHTML = s.sentence.map((w, i) => {
        const isNew = w === s.last_added &&
                      i === s.sentence.length - 1 &&
                      w !== prevLastAdded;
        return `<span class="word ${isNew ? 'new' : ''}">${w.toUpperCase()}</span> `;
      }).join('');
    }

    prevLastAdded = s.last_added;
    prevSentence  = [...(s.sentence || [])];
  }

  async function fetchState() {
    try {
      const r = await fetch('/state');
      const s = await r.json();
      updateUI(s);
    } catch(e) {}
  }

  async function clearSentence() {
    const text = prevSentence.join(' ');
    if (text.trim()) {
      history.unshift({ text, time: new Date().toLocaleTimeString() });
      updateHistory();
    }
    await fetch('/clear', { method: 'POST' });
  }

  async function undoLast() {
    await fetch('/undo', { method: 'POST' });
  }

  function speakSentence() {
    const speakMap = { "Fck u": "fuck you" };
    const text = prevSentence.map(w => speakMap[w] || w).join(' ');
    if ('speechSynthesis' in window && text) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.9;
      speechSynthesis.speak(u);
    }
  }

  async function shutdown() {
    if (confirm("Stop the server and release the camera?")) {
      await fetch('/shutdown', { method: 'POST' });
      document.getElementById('shutdownOverlay').classList.add('show');
    }
  }

  function updateHistory() {
    const el = document.getElementById('historyList');
    if (!history.length) {
      el.innerHTML = '<div class="empty-state">Completed sentences appear here</div>';
      return;
    }
    el.innerHTML = history.slice(0, 10).map(h => `
      <div class="history-item">
        <span>${h.text}</span>
        <span class="ts">${h.time}</span>
      </div>
    `).join('');
  }

  setInterval(fetchState, 120);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("=" * 50)
    print("  SIGN LANGUAGE FLASK DASHBOARD")
    print("=" * 50)
    print("  Open: http://localhost:5000")
    print("  Stop: click 'Stop Server' button in browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)