import cv2
import mediapipe as mp
import time
import subprocess
import pyautogui
import math
import json

# Utility to compute distance between two 2D points
print('afasdf')
def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load or create gesture-to-hotkey mapping
DEFAULT_MAPPING = {
    "Fist": "ctrl,down",
    "Open": "ctrl,up",
    "Victory": "ctrl,command,f",
    "Pointing": "ctrl,right",
    "OK": "command,shift,3"
}

CONFIG_PATH = 'gesture_hotkeys.json'

try:
    with open(CONFIG_PATH, 'r') as f:
        mapping = json.load(f)
        print(f"Loaded custom mapping from {CONFIG_PATH}")
except FileNotFoundError:
    mapping = DEFAULT_MAPPING.copy()
    print("No config found; using default mappings.")

# Allow user to customize mappings interactively
print("Configure hotkeys for each gesture (comma-separated modifiers+key). Leave blank to keep current.")
for gesture, combo in mapping.items():
    user_input = input(f"{gesture} [{combo}]: ").strip()
    if user_input:
        mapping[gesture] = user_input

# Save updated config
with open(CONFIG_PATH, 'w') as f:
    json.dump(mapping, f, indent=2)
    print(f"Saved custom mapping to {CONFIG_PATH}")

# Video capture start
cap = cv2.VideoCapture(0)
prev_gesture = None
prev_change_time = None
THRESHOLD_TIME = 0.9

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    current_gesture = "No hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
        tips = [lm[i] for i in (4, 8, 12, 16, 20)]
        pips = [lm[i] for i in (3, 6, 10, 14, 18)]
        thumb_extended = tips[0][0] > pips[0][0]
        fingers_extended = [(tips[i][1] < pips[i][1]) for i in range(1,5)]
        idx, mid, ring, pinky = fingers_extended

        ok_dist = distance(lm[4], lm[8])
        if idx and mid and not ring and not pinky:
            current_gesture = "Victory"
        elif all(fingers_extended):
            current_gesture = "Open"
        elif not any(fingers_extended) and not thumb_extended:
            current_gesture = "Fist"
        elif idx and pinky and not (mid or ring or thumb_extended):
            current_gesture = "Pointing"
        elif ok_dist < 0.05 and all(fingers_extended[1:]):
            current_gesture = "OK"
        else:
            current_gesture = "Fist"

    now = time.time()
    # On gesture change and within threshold, trigger hotkey
    if prev_gesture is None:
        prev_gesture = current_gesture
        prev_change_time = now
    elif current_gesture != prev_gesture:
        dt = now - prev_change_time
        if dt <= THRESHOLD_TIME and current_gesture in mapping:
            combo = mapping[current_gesture].split(',')
            pyautogui.hotkey(*combo)
        prev_gesture = current_gesture
        prev_change_time = now

    # Display
    cv2.putText(image, f"Gesture: {current_gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Classifier (Mirrored)', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
