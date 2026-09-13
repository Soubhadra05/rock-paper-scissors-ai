"""
Rock Paper Scissors vs AI — camera-controlled edition
------------------------------------------------------
Show your hand gesture to the webcam. The AI picks (or counter-picks
you, on Hard difficulty). Gesture is classified by counting extended
fingers via MediaPipe hand landmarks — no trained model needed.

Built on MediaPipe's current Tasks API (HandLandmarker). Includes a
low-light boost pipeline and multi-frame vote capture so detection
stays reliable even in dim rooms or with a shaky hand.

Controls
--------
Menu:
  1 / 3 / 5 / 7   choose match length (best of N)
  e / d           choose difficulty (Easy / haRd)
  t               practice mode (test gesture detection, no scoring)
  SPACE           start the match
During a match:
  SPACE           start a round
  m               abandon match, back to menu
After a match:
  SPACE           rematch (same settings)
  m               back to menu
Practice mode:
  m               back to menu
Any time:
  i               toggle the help overlay
  p               pause / resume
  s               mute / unmute sound
  q               quit
"""

import json
import os
import random
import time
import urllib.request
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import sounddevice as sd
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model setup — auto-download the hand landmark model on first run
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
STATS_PATH = os.path.join(SCRIPT_DIR, "stats.json")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, a few MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


# ---------------------------------------------------------------------------
# Persistent all-time stats
# ---------------------------------------------------------------------------

DEFAULT_STATS = {
    "wins": 0, "losses": 0, "draws": 0,
    "matches_played": 0, "matches_won": 0, "best_streak": 0,
}


def load_stats():
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH) as f:
                data = json.load(f)
            merged = DEFAULT_STATS.copy()
            merged.update(data)
            return merged
        except Exception:
            pass
    return DEFAULT_STATS.copy()


def save_stats(stats):
    try:
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f)
    except Exception:
        pass  # non-fatal — game keeps working without a writable folder


# ---------------------------------------------------------------------------
# Sound effects (generated tones, no audio files needed)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
SOUND_ENABLED = True


def _tone(freq, duration, volume=0.25):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = volume * np.sin(freq * t * 2 * np.pi)
    fade = min(200, len(wave) // 4)
    if fade > 0:
        envelope = np.ones(len(wave))
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        wave = wave * envelope
    return wave.astype(np.float32)


def _chime(freqs, duration=0.1):
    return np.concatenate([_tone(f, duration) for f in freqs])


SOUNDS = {
    "tick": _tone(700, 0.08),
    "shoot": _tone(1300, 0.18),
    "win": _chime([523, 659, 784]),     # ascending major arpeggio
    "lose": _chime([392, 311, 233]),    # descending
    "draw": _tone(440, 0.2),
    "select": _tone(900, 0.05),
}


def play(name):
    if not (SOUND_AVAILABLE and SOUND_ENABLED):
        return
    try:
        sd.play(SOUNDS[name], samplerate=SAMPLE_RATE)
    except Exception:
        pass  # no audio device available — fail silently, game still works


# ---------------------------------------------------------------------------
# Hand skeleton drawing
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


# ---------------------------------------------------------------------------
# Low-light enhancement
# ---------------------------------------------------------------------------

LOW_LIGHT_THRESHOLD = 95
GAMMA = 1.6
_gamma_lut = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]
).astype("uint8")


def enhance_low_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= LOW_LIGHT_THRESHOLD:
        return frame, False
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    enhanced = cv2.LUT(enhanced, _gamma_lut)
    return enhanced, True


# ---------------------------------------------------------------------------
# Gesture classification (thumb excluded — see note below)
# ---------------------------------------------------------------------------
#
# The thumb's "extended" state depends heavily on hand rotation relative
# to the camera, which made it unreliable. Since Rock needs exactly 0
# fingers and Scissors needs exactly 2, a single thumb misread was enough
# to throw off classification. Rock/Paper/Scissors are fully distinguishable
# using just the other four fingers.

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_MCPS = [5, 9, 13, 17]

MOVES = ["Rock", "Paper", "Scissors"]
BEATS = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}


def get_extended_fingers(landmarks):
    extended = []
    for tip, pip, mcp in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS):
        is_up = landmarks[tip].y < landmarks[pip].y and landmarks[tip].y < landmarks[mcp].y
        extended.append(is_up)
    return extended


def classify_gesture(extended):
    index, middle, ring, pinky = extended
    count = sum(extended)
    if count == 0:
        return "Rock"
    if count == 4:
        return "Paper"
    if index and middle and not ring and not pinky:
        return "Scissors"
    return None


def decide_winner(player, ai):
    if player == ai:
        return "Draw"
    if BEATS.get(player) == ai:
        return "You Win!"
    return "AI Wins!"


def ai_choose(difficulty, player_history):
    if difficulty == "Hard" and len(player_history) >= 3 and random.random() < 0.7:
        most_common = Counter(player_history).most_common(1)[0][0]
        counter = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
        return counter[most_common]
    return random.choice(MOVES)


# ---------------------------------------------------------------------------
# Gesture icons, countdown ring, confetti, achievements
# ---------------------------------------------------------------------------

def draw_gesture_icon(frame, move, center, radius=45, color=(255, 255, 255)):
    x, y = center
    if move == "Rock":
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (0, 0, 0), 2)
    elif move == "Paper":
        cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius), color, -1)
        cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 0), 2)
    elif move == "Scissors":
        cv2.circle(frame, (x, y), radius, (40, 40, 40), 2)
        off = int(radius * 0.7)
        cv2.line(frame, (x - off, y - off), (x + off, y + off), color, 8)
        cv2.line(frame, (x - off, y + off), (x + off, y - off), color, 8)
    else:
        cv2.circle(frame, (x, y), radius, (60, 60, 60), 2)
        cv2.putText(frame, "?", (x - 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def draw_countdown_ring(frame, fraction, center, radius=70):
    cv2.circle(frame, center, radius, (60, 60, 60), 4)
    angle = 360 * max(0.0, min(1.0, fraction))
    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 0, 255), 8)


def spawn_confetti(width):
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)]
    return [
        {
            "x": random.randint(0, width),
            "y": random.randint(-60, 0),
            "vx": random.uniform(-2, 2),
            "vy": random.uniform(2, 5),
            "life": random.randint(25, 45),
            "color": random.choice(colors),
        }
        for _ in range(40)
    ]


def update_confetti(frame, particles):
    alive = []
    h = frame.shape[0]
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.15
        p["life"] -= 1
        if p["life"] > 0 and p["y"] < h:
            cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, p["color"], -1)
            alive.append(p)
    return alive


ACHIEVEMENT_SECONDS = 2.5


def add_achievement(achievements, text):
    achievements.append({"text": text, "until": time.time() + ACHIEVEMENT_SECONDS})


def draw_achievements(frame, achievements):
    w = frame.shape[1]
    y = 60
    for item in achievements:
        text = item["text"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = w // 2 - tw // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 15, y - th - 12), (x + tw + 15, y + 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
        y += 45


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def draw_translucent_box(frame, x1, y1, x2, y2, alpha=0.55, color=(20, 20, 20)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_help_overlay(frame, lines):
    h, w = frame.shape[:2]
    box_h = 30 + 28 * len(lines)
    draw_translucent_box(frame, 20, h // 2 - box_h // 2, w - 20, h // 2 + box_h // 2)
    y = h // 2 - box_h // 2 + 30
    for line in lines:
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 28


HELP_LINES = [
    "SPACE - start round / start match / rematch",
    "m - back to menu (match, match-over, practice)",
    "1 / 3 / 5 / 7 - best-of length (in menu)",
    "e / d - Easy / harD difficulty (in menu)",
    "t - practice mode (in menu)",
    "p - pause / resume     s - mute / unmute sound",
    "i - toggle this help     q - quit",
]


# ---------------------------------------------------------------------------
# Game state machine
# ---------------------------------------------------------------------------

MENU, WAITING, COUNTDOWN, CAPTURE, RETRY, RESULT, MATCH_OVER, PRACTICE = (
    "MENU", "WAITING", "COUNTDOWN", "CAPTURE", "RETRY", "RESULT", "MATCH_OVER", "PRACTICE",
)

COUNTDOWN_SECONDS = 3
CAPTURE_WINDOW = 0.6
RETRY_DISPLAY_SECONDS = 1.3
RESULT_DISPLAY_SECONDS = 2.2
FLASH_SECONDS = 0.35


def main():
    ensure_model()
    stats = load_stats()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    best_of = 3
    difficulty = "Easy"

    def new_match():
        return {
            "score": {"You": 0, "AI": 0},
            "round_log": [],
            "history": [],
            "streak_owner": None,
            "streak_len": 0,
        }

    match = new_match()

    global SOUND_ENABLED

    state = MENU
    countdown_start = 0.0
    countdown_last_label = None
    capture_start = 0.0
    capture_votes = []
    retry_start = 0.0
    result_start = 0.0
    player_move = None
    ai_move = None
    outcome = None
    match_winner = None
    flash_color = None
    flash_until = 0.0
    show_help = False
    paused = False
    gesture_buffer = deque(maxlen=7)
    confetti = []
    achievements = []

    print("Press SPACE in-app to navigate. See on-screen instructions ('i').")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame, boosted = enhance_low_light(frame)

        if paused:
            h, w = frame.shape[:2]
            draw_translucent_box(frame, 0, 0, w, h, alpha=0.5)
            cv2.putText(frame, "PAUSED", (w // 2 - 90, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(frame, "press p to resume", (w // 2 - 110, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("Rock Paper Scissors vs AI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = False
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        live_gesture = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_hand(frame, hand_landmarks)
            extended_fingers = get_extended_fingers(hand_landmarks)
            live_gesture = classify_gesture(extended_fingers)

        gesture_buffer.append(live_gesture)
        h, w = frame.shape[:2]

        if boosted:
            cv2.putText(frame, "Low-light boost: ON", (w - 260, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        if not SOUND_ENABLED:
            cv2.putText(frame, "Muted", (w - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # ------------------------------------------------------------- MENU
        if state == MENU:
            cv2.putText(frame, "ROCK  PAPER  SCISSORS  vs AI", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Best of: {best_of}  (press 1 / 3 / 5 / 7)", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Difficulty: {difficulty}  (press e / d)", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press t for practice mode", (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            cv2.putText(frame, "Press SPACE to start", (30, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"All-time  W {stats['wins']}  L {stats['losses']}  D {stats['draws']}   "
                f"Matches won {stats['matches_won']}/{stats['matches_played']}   "
                f"Best streak {stats['best_streak']}",
                (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
            )

        # -------------------------------------------------------- PRACTICE
        elif state == PRACTICE:
            cv2.putText(frame, "PRACTICE MODE - m to return to menu", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            valid_votes = [g for g in gesture_buffer if g]
            steady_gesture = Counter(valid_votes).most_common(1)[0][0] if valid_votes else None
            confidence = (valid_votes.count(steady_gesture) / len(gesture_buffer)) if steady_gesture else 0
            draw_gesture_icon(frame, steady_gesture, (w - 90, 110), radius=55)
            cv2.putText(frame, steady_gesture or "No gesture detected", (30, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if steady_gesture else (0, 0, 255), 2)
            bar_w = int(200 * confidence)
            cv2.rectangle(frame, (30, h - 35), (230, h - 20), (80, 80, 80), 1)
            cv2.rectangle(frame, (30, h - 35), (30 + bar_w, h - 20), (0, 255, 0), -1)

        # ---------------------------------------------------------- WAITING
        elif state == WAITING:
            cv2.putText(frame, "Press SPACE to play a round", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            valid_votes = [g for g in gesture_buffer if g]
            steady_gesture = Counter(valid_votes).most_common(1)[0][0] if valid_votes else None
            if steady_gesture:
                cv2.putText(frame, f"Detected: {steady_gesture}", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                draw_gesture_icon(frame, steady_gesture, (w - 60, 60), radius=30)

        # -------------------------------------------------------- COUNTDOWN
        elif state == COUNTDOWN:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            label = str(remaining) if remaining > 0 else "SHOOT!"
            if label != countdown_last_label:
                play("shoot" if label == "SHOOT!" else "tick")
                countdown_last_label = label
            center = (w // 2, h // 2)
            fraction = max(0.0, (COUNTDOWN_SECONDS - elapsed) / COUNTDOWN_SECONDS)
            draw_countdown_ring(frame, fraction, center)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]
            cv2.putText(frame, label, (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            if elapsed >= COUNTDOWN_SECONDS:
                state = CAPTURE
                capture_start = time.time()
                capture_votes = []

        # ---------------------------------------------------------- CAPTURE
        elif state == CAPTURE:
            if live_gesture:
                capture_votes.append(live_gesture)
            cv2.putText(frame, "Hold your gesture...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if time.time() - capture_start >= CAPTURE_WINDOW:
                if capture_votes:
                    player_move = Counter(capture_votes).most_common(1)[0][0]
                    ai_move = ai_choose(difficulty, match["history"])
                    match["history"].append(player_move)
                    outcome = decide_winner(player_move, ai_move)

                    if outcome == "You Win!":
                        match["score"]["You"] += 1
                        match["round_log"].append("W")
                        flash_color = (0, 200, 0)
                        stats["wins"] += 1
                        if match["streak_owner"] == "You":
                            match["streak_len"] += 1
                        else:
                            match["streak_owner"], match["streak_len"] = "You", 1
                        stats["best_streak"] = max(stats["best_streak"], match["streak_len"])
                        confetti = spawn_confetti(w)
                        play("win")
                        if stats["wins"] == 1:
                            add_achievement(achievements, "ACHIEVEMENT: First Win!")
                        if match["streak_len"] == 3:
                            add_achievement(achievements, "ACHIEVEMENT: 3-Win Streak!")
                        elif match["streak_len"] == 5:
                            add_achievement(achievements, "ACHIEVEMENT: 5-Win Streak - On Fire!")
                    elif outcome == "AI Wins!":
                        match["score"]["AI"] += 1
                        match["round_log"].append("L")
                        flash_color = (0, 0, 200)
                        stats["losses"] += 1
                        if match["streak_owner"] == "AI":
                            match["streak_len"] += 1
                        else:
                            match["streak_owner"], match["streak_len"] = "AI", 1
                        play("lose")
                    else:
                        match["round_log"].append("D")
                        flash_color = (0, 200, 200)
                        stats["draws"] += 1
                        match["streak_owner"], match["streak_len"] = None, 0
                        play("draw")

                    save_stats(stats)
                    flash_until = time.time() + FLASH_SECONDS
                    result_start = time.time()
                    state = RESULT

                    wins_needed = best_of // 2 + 1
                    if match["score"]["You"] >= wins_needed:
                        match_winner = "You"
                    elif match["score"]["AI"] >= wins_needed:
                        match_winner = "AI"
                else:
                    retry_start = time.time()
                    state = RETRY

        # ------------------------------------------------------------ RETRY
        elif state == RETRY:
            cv2.putText(frame, "Couldn't see your hand clearly.", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Hold it closer, steadier, and in the light.", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            if time.time() - retry_start >= RETRY_DISPLAY_SECONDS:
                state = COUNTDOWN
                countdown_start = time.time()
                countdown_last_label = None

        # ----------------------------------------------------------- RESULT
        elif state == RESULT:
            draw_gesture_icon(frame, player_move, (w // 2 - 100, 110), radius=50, color=(255, 220, 0))
            draw_gesture_icon(frame, ai_move, (w // 2 + 100, 110), radius=50, color=(0, 165, 255))
            cv2.putText(frame, "You", (w // 2 - 118, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
            cv2.putText(frame, "AI", (w // 2 + 88, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            text_size = cv2.getTextSize(outcome, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            cv2.putText(frame, outcome, (w // 2 - text_size[0] // 2, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if match["streak_len"] >= 2:
                cv2.putText(frame, f"{match['streak_owner']} streak x{match['streak_len']}",
                            (w // 2 - 90, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
            if time.time() - result_start >= RESULT_DISPLAY_SECONDS:
                if match_winner:
                    stats["matches_played"] += 1
                    if match_winner == "You":
                        stats["matches_won"] += 1
                        if match["score"]["AI"] == 0:
                            add_achievement(achievements, "ACHIEVEMENT: Flawless Victory!")
                    save_stats(stats)
                    state = MATCH_OVER
                else:
                    state = WAITING

        # ------------------------------------------------------- MATCH_OVER
        elif state == MATCH_OVER:
            cv2.putText(frame, f"{match_winner} WON THE MATCH!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if match_winner == "You" else (0, 0, 255), 3)
            cv2.putText(frame, f"Final score  You {match['score']['You']} - {match['score']['AI']} AI",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE = rematch    m = menu", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ------------------------------------------------ always-on widgets
        if state not in (MENU, PRACTICE):
            cv2.putText(
                frame,
                f"You {match['score']['You']}  -  AI {match['score']['AI']}   (best of {best_of}, {difficulty})",
                (30, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            if match["round_log"]:
                log_text = "Rounds: " + " ".join(match["round_log"])
                cv2.putText(frame, log_text, (30, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if flash_color and time.time() < flash_until:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), flash_color, 12)

        confetti = update_confetti(frame, confetti)
        achievements = [a for a in achievements if time.time() < a["until"]]
        draw_achievements(frame, achievements)

        if show_help:
            draw_help_overlay(frame, HELP_LINES)

        cv2.imshow("Rock Paper Scissors vs AI", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('i'):
            show_help = not show_help
        elif key == ord('p'):
            paused = True
        elif key == ord('s'):
            SOUND_ENABLED = not SOUND_ENABLED
        elif state == MENU:
            if key in (ord('1'), ord('3'), ord('5'), ord('7')):
                best_of = int(chr(key))
                play("select")
            elif key == ord('e'):
                difficulty = "Easy"
                play("select")
            elif key == ord('d'):
                difficulty = "Hard"
                play("select")
            elif key == ord('t'):
                state = PRACTICE
            elif key == ord(' '):
                match = new_match()
                match_winner = None
                state = WAITING
        elif state == PRACTICE and key == ord('m'):
            state = MENU
        elif state == WAITING and key == ord(' '):
            state = COUNTDOWN
            countdown_start = time.time()
            countdown_last_label = None
        elif state == WAITING and key == ord('m'):
            state = MENU
        elif state == MATCH_OVER and key == ord(' '):
            match = new_match()
            match_winner = None
            state = WAITING
        elif state == MATCH_OVER and key == ord('m'):
            state = MENU

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
