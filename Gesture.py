import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller, Button
import keyboard
from pynput.keyboard import Key, Controller as ctl
# Constants
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
CLICK_THRESHOLD = 20
ERASE_THRESHOLD = 30
WAVE_THRESHOLD = 100
BLACKBOARD_OVERLAY_WEIGHT = 0.3
SELFIE_HOLD_TIME = 2
FINGER_CLOSE_THRESHOLD = 30
ZOOM_IN_THRESHOLD = 200
ZOOM_OUT_THRESHOLD = 100

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

mouse = Controller()
kb =ctl()
current_time = time.time()

control_mode = "volume"  # Default mode
last_switch_time = 0  # Prevent rapid switching

def switch_control_mode():
    global control_mode, last_switch_time
    if time.time() - last_switch_time > 1:  # Prevent frequent switching
        control_mode = "brightness" if control_mode == "volume" else "volume"
        print(f"Switched to {control_mode} mode")
        last_switch_time = time.time()

# Initialize camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get camera frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame.")
    exit()
h, w, _ = frame.shape

# Create a blackboard for drawing
blackboard = np.zeros((h, w, 3), dtype=np.uint8)
prev_x, prev_y = None, None
start_time_right_palm = None
start_time_left_palm = None

drawing = False
volume_control = False
brightness_control = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?).")
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Extract key points
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)
            ring_x, ring_y = int(landmarks[16].x * w), int(landmarks[16].y * h)
            pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)

            # Mouse movement using index & middle fingers
            if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:
                mouse.position = (index_x, index_y)

            # Left Click: Index & thumb contact
            if np.hypot(index_x - thumb_x, index_y - thumb_y) < CLICK_THRESHOLD:
                mouse.click(Button.left, 1)
                time.sleep(0.2)

            # Right Click: Middle & thumb contact
            if np.hypot(middle_x - thumb_x, middle_y - thumb_y) < CLICK_THRESHOLD:
                mouse.click(Button.right, 1)
                time.sleep(0.2)

            # Scroll with three fingers
            if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y:
                mouse.scroll(0, -1 if index_y < h // 2 else 1)
                time.sleep(0.2)

            # Drawing Mode: Index finger up
            if landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                drawing = True
                if prev_x is not None and prev_y is not None:
                    cv.line(blackboard, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 5)
                prev_x, prev_y = index_x, index_y
            else:
                drawing = False
                prev_x, prev_y = None, None

            # Erase: Thumb & index contact
            if np.hypot(index_x - thumb_x, index_y - thumb_y) < ERASE_THRESHOLD:
                cv.circle(blackboard, (index_x, index_y), 20, (0, 0, 0), -1)

            # Wave to clear the screen
            if abs(index_x - pinky_x) > WAVE_THRESHOLD:
                blackboard.fill(0)

            

            # Two fingers for volume/brightness control
            if np.hypot(index_x - middle_x, index_y - middle_y) < FINGER_CLOSE_THRESHOLD:  # Fingers close
                if volume_control:
                    cv.putText(frame, "Volume Control", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if index_y < h // 2:
                        keyboard.press_and_release('volumeup')
                    else:
                        keyboard.press_and_release('volumedown')
                elif brightness_control:
                    cv.putText(frame, "Brightness Control", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if index_y < h // 2:
                        kb.press('brightnessup')
                    else:
                        kb.release('brightnessdown')

                time.sleep(0.2)
            # Switch between volume & brightness with little finger & thumb touch
            if np.hypot(pinky_x - thumb_x, pinky_y - thumb_y) < CLICK_THRESHOLD:
                volume_control, brightness_control = brightness_control, volume_control
                time.sleep(0.3)

            

    frame = cv.addWeighted(frame, 1 - BLACKBOARD_OVERLAY_WEIGHT, blackboard, BLACKBOARD_OVERLAY_WEIGHT, 0)
    cv.imshow("Hand Gesture Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()