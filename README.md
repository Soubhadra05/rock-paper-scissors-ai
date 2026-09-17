# Rock Paper Scissors vs AI — Optimized

This build improves real-time hand detection speed and stability while keeping the original webcam RPS gameplay.

## Detection improvements
- MediaPipe HandLandmarker VIDEO mode with temporal tracking.
- Detection input is downscaled to 640px wide to reduce CPU work.
- Detection is capped at ~30 FPS while the camera/UI can run faster.
- Camera buffer is set to 1 frame to reduce stale-frame latency.
- Finger classification uses joint angles and normalized landmark distances instead of only vertical Y comparisons.
- Confidence-weighted temporal smoothing rejects unstable gestures.
- Low-light processing is adaptive: cheap gamma correction first, CLAHE only for very dark frames.
- Capture window reduced from 0.6s to 0.32s.

## IMPORTANT: Windows OpenCV fix

If you see an error mentioning:

`cv2.error: ... The function is not implemented ... cv2.imshow`

your Python environment has a headless OpenCV build or an OpenCV 5 build without the desktop HighGUI backend.

### Recommended setup

1. Open this project folder in VS Code.
2. Double-click **`setup_windows.bat`** or run it from the VS Code terminal:

```bat
setup_windows.bat
```

The script removes conflicting OpenCV/headless packages and installs the tested desktop build:

`opencv-python==4.11.0.86`

3. Start the game:

```bash
python main.py
```

### Manual setup

If you prefer the terminal:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install -r requirements.txt
python main.py
```

The MediaPipe model is downloaded automatically on first run if `hand_landmarker.task` is missing.

## Controls
- Menu: `1/3/5/7` match length, `E/D` difficulty, `T` practice, `SPACE` start
- During match: `SPACE` round, `M` menu
- Any screen: `P` pause, `S` mute, `Q` quit

## Crash fix in this build
- Fixed a match-start crash caused by the game calling `engine.reset_smoothing()` without defining that method.
- Gesture smoothing is now explicitly reset at menu/round transitions so previous-round detections cannot leak into the next round.
- Added safer MediaPipe initialization and transient webcam-frame retry handling.
