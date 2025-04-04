import time
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import platform
import subprocess

try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

# Constants
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
CLICK_THRESHOLD = 20
ERASE_THRESHOLD = 30
WAVE_THRESHOLD = 100
FINGER_CLOSE_THRESHOLD = 40  # Slightly increased for better accuracy

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

# Get screen size
control_mode = "volume"
last_switch_time = 0
screen_w, screen_h = pyautogui.size()

# Initialize camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame.")
    exit()

h, w, _ = frame.shape  # Get frame height and width
blackboard = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None

def switch_control_mode(landmarks):
    """ Switches between volume and brightness using thumb and pinky """
    global control_mode, last_switch_time
    if time.time() - last_switch_time > 1:  # Prevents frequent switching
        thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
        pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)

        if abs(thumb_x - pinky_x) < FINGER_CLOSE_THRESHOLD and abs(thumb_y - pinky_y) < FINGER_CLOSE_THRESHOLD:
            control_mode = "brightness" if control_mode == "volume" else "volume"
            last_switch_time = time.time()
            print(f"Switched to: {control_mode}")

def change_brightness(increase=True, percent=10):
    system_platform = platform.system().lower()
    
    try:
        if system_platform == 'windows' and sbc:
            current = sbc.get_brightness(display=0)[0]
            new_brightness = min(100, current + percent) if increase else max(0, current - percent)
            sbc.set_brightness(new_brightness, display=0)

        elif system_platform == 'linux':
            cmd = f"xbacklight -{'inc' if increase else 'dec'} {percent}"
            subprocess.run(cmd, shell=True)

        elif system_platform == 'darwin':  # macOS
            # Requires `brightness` command-line tool
            print("macOS brightness control not implemented. Use external tool like 'brightness'.")

        else:
            print(f"Unsupported platform: {system_platform}")

    except Exception as e:
        print(f"Brightness control failed: {e}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)

            # Convert hand coordinates to screen coordinates
            screen_x = np.interp(index_x, (0, w), (0, screen_w))
            screen_y = np.interp(index_y, (0, h), (0, screen_h))

            action_done = False  # Prevent multiple actions per frame

            # Right Hand: Mouse Control & Drawing
            if hand_label == "Right":
                if not action_done and np.hypot(index_x - thumb_x, index_y - thumb_y) < CLICK_THRESHOLD:
                    pyautogui.click()
                    action_done = True

                elif not action_done and np.hypot(index_x - pinky_x, index_y - pinky_y) < CLICK_THRESHOLD:
                    pyautogui.rightClick()
                    action_done = True

                elif not action_done and all(landmarks[finger].y < landmarks[finger - 2].y for finger in [8, 12, 16]):
                    pyautogui.scroll(-10 if index_y < h // 2 else 10)  # Reduced delay by increasing scroll speed
                    action_done = True

                elif not action_done and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                    if prev_x is not None and prev_y is not None:
                        cv.line(blackboard, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 5)
                    prev_x, prev_y = index_x, index_y
                    action_done = True
                else:
                    prev_x, prev_y = None, None

                if not action_done and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Faster response

            # Left Hand: Volume/Brightness Control
            elif hand_label == "Left":
                if not action_done and np.hypot(index_x - thumb_x, index_y - thumb_y) < ERASE_THRESHOLD:
                    cv.circle(blackboard, (index_x, index_y), 20, (0, 0, 0), -1)
                    action_done = True

                elif not action_done and abs(index_x - pinky_x) > WAVE_THRESHOLD:
                    blackboard.fill(0)
                    action_done = True

                # Switch mode
                switch_control_mode(landmarks)

                # Adjust volume/brightness
                if control_mode == "volume":
                    if index_y < h // 2:
                        pyautogui.press('volumeup')
                    else:
                        pyautogui.press('volumedown')
                else:  # Brightness mode
                    if index_y < h // 2:
                        change_brightness(increase=True)
                    else:
                        change_brightness(increase=False)

    # Overlay BlackBoard
    frame = cv.addWeighted(frame, 1, blackboard, 0.3, 0)
    cv.imshow("Hand Gesture Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
