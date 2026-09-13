# Rock Paper Scissors vs AI (Camera-Controlled)

A full-featured webcam Rock-Paper-Scissors game: sound effects, gesture
icons, animated countdown, win confetti, achievements, persistent
all-time stats, a practice mode, and low-light-tolerant hand detection.

Built on MediaPipe's current **Tasks API** (`HandLandmarker`). Note for
anyone following older tutorials: MediaPipe 0.11+ removed the old
`mp.solutions.hands` API entirely — if you see
`AttributeError: module 'mediapipe' has no attribute 'solutions'`, the
code you're looking at is outdated, not your install.

## How detection works

1. **Hand landmarks** — MediaPipe's `HandLandmarker` locates 21 points
   on your hand each frame (model auto-downloads on first run).
2. **Gesture classification** — counts only the four non-thumb fingers
   (the thumb's extended/curled state is unreliable across hand angles,
   so it's excluded entirely):
   - 0 of 4 fingers up → **Rock**
   - Index + middle only → **Scissors**
   - All 4 up → **Paper**
3. **Low-light boost** — if the frame is dim, CLAHE contrast enhancement
   (on the LAB lightness channel) plus gamma correction runs automatically
   before detection. A "Low-light boost: ON" label shows when active.
4. **Vote-based capture** — the "Shoot!" moment captures ~0.6 seconds of
   frames and takes the majority-vote gesture, rather than trusting one
   possibly-blurry frame. An unreadable gesture triggers a retry instead
   of silently miscounting.

## Gameplay features

- **Best-of-N matches** (1 / 3 / 5 / 7), chosen from the menu.
- **Two AI difficulties** — Easy (random) or Hard (tracks your move
  history and counter-picks your most frequent move ~70% of the time).
- **Gesture icons** — Rock/Paper/Scissors render as actual shapes (a
  circle, a square, a scissor-cross), not just text, both for your live
  detected gesture and the round result reveal.
- **Animated countdown ring** around the 3-2-1 numbers.
- **Sound effects** — countdown ticks, a "shoot" cue, and distinct
  win/lose/draw chimes, generated on the fly (no audio files needed).
  Mute anytime with `s`.
- **Confetti burst** on every round you win, plus a colored screen-flash
  (green/red/yellow) for win/lose/draw.
- **Win streak tracking** and **achievement pop-ups** (first win, 3- and
  5-win streaks, a flawless-victory match).
- **Persistent all-time stats** (wins/losses/draws, matches won, best
  streak) saved to `stats.json` next to the script, shown on the menu.
- **Practice mode** (`t` from the menu) — see your live detected gesture
  and a confidence bar with no scoring, to dial in lighting/hand position
  before playing a real match.
- **Pause** (`p`) anytime.
- **Retry instead of miscount** — an unreadable gesture re-runs the round.
- **In-game help overlay** (`i`) listing all controls.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Requires a working webcam and an internet connection on first run (to
download the ~10 MB hand landmark model). If `pip install` fails on your
Python version, try Python 3.10 or 3.11 — MediaPipe doesn't support
every latest Python release on day one. Sound effects use `sounddevice`;
if no audio output device is available, the game keeps working silently
(sound calls fail safe).

## Controls

| Screen        | Key             | Action                              |
|---------------|-----------------|--------------------------------------|
| Menu          | `1` `3` `5` `7` | Set match length (best of N)        |
| Menu          | `e` / `d`       | Easy / harD difficulty               |
| Menu          | `t`             | Enter practice mode                  |
| Menu          | `SPACE`         | Start the match                      |
| In match      | `SPACE`         | Start a round                        |
| In match      | `m`             | Abandon match, back to menu          |
| Match over    | `SPACE`         | Rematch (same settings)              |
| Match over    | `m`             | Back to menu                         |
| Practice      | `m`             | Back to menu                         |
| Any screen    | `i`             | Toggle help overlay                  |
| Any screen    | `p`             | Pause / resume                       |
| Any screen    | `s`             | Mute / unmute sound                  |
| Any screen    | `q`             | Quit                                  |

## Ideas to extend it further (good for a project report's "future scope")

- **Two-player mode**: detect two hands and skip the AI entirely.
- **Replace rule-based classification with a trained CNN** on a
  hand-gesture dataset — a good talking point if your course wants to
  see an ML model rather than geometric rules.
- **Web version**: port to a browser using MediaPipe's JS/WASM build +
  TensorFlow.js so it runs without a Python install.
- **Online leaderboard**: sync `stats.json` to a small backend so
  friends can compare records.
- **More achievements**: comeback wins, perfect practice-mode accuracy,
  longest session played.

## Project structure

```
rps_ai/
├── main.py           # game logic + camera loop
├── requirements.txt
├── stats.json         # created automatically after your first round
└── README.md
```
