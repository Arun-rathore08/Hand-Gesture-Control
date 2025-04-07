import os
import cv2 as cv
import time
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller, Button
import platform
import pyautogui
import shutil

# Constants
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
CLICK_THRESHOLD = 20
FINGER_CLOSE_THRESHOLD = 30
CONTROL_SWITCH_COOLDOWN = 1.0
BLACKBOARD_OVERLAY_WEIGHT = 0.1
WAVE_THRESHOLD = 100
SCROLL_SENSITIVITY = 25  # Adjust for scroll speed

last_wave_time = 0
wave_gesture_active = False
wave_positions = []
MUTE_COOLDOWN = 2.0  # seconds

# Initialize previous finger positions for scrolling
prev_finger_positions = {"index": None, "middle": None, "ring": None}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

mouse = Controller()
last_switch_time = 0
control_mode = "volume"
OS_TYPE = platform.system()

# Initialize camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame.")
    exit()
h, w, _ = frame.shape

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Create a blackboard
blackboard = np.zeros((h, w, 3), dtype=np.uint8)
prev_index_x, prev_index_y = None, None

def toggle_mute():
    try:
        os.system("amixer -D pulse set Master toggle > /dev/null")
    except Exception as e:
        print(f"Error toggling mute: {e}")


# Volume and Brightness Control Functions
def increase_volume():
    try:
        if OS_TYPE == "Linux":
            os.system("amixer -D pulse sset Master 5%+ > /dev/null")
        elif OS_TYPE == "Windows":
            # You can use nircmd for Windows: https://www.nirsoft.net/utils/nircmd.html
            pyautogui.press('volumeup')
        elif OS_TYPE == "Darwin":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 5)'")
        else:
            print("Unsupported OS for volume control.")
    except Exception as e:
        print(f"Error increasing volume: {e}")

def decrease_volume():
    try:
        if OS_TYPE == "Linux":
            os.system("amixer -D pulse sset Master 5%- > /dev/null")
        elif OS_TYPE == "Windows":
            pyautogui.press('volumedown')
        elif OS_TYPE == "Darwin":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 5)'")
        else:
            print("Unsupported OS for volume control.")
    except Exception as e:
        print(f"Error decreasing volume: {e}")

def increase_brightness(step=10):
    system = platform.system()

    if system == "Linux":
        # Try sysfs first
        try:
            path = "/sys/class/backlight/intel_backlight"
            with open(f"{path}/brightness", "r") as f:
                current = int(f.read().strip())
            with open(f"{path}/max_brightness", "r") as f:
                max_val = int(f.read().strip())

            new = min(current + int((step / 100) * max_val), max_val)
            with open(f"{path}/brightness", "w") as f:
                f.write(str(new))
            return
        except PermissionError:
            print("Run as root to change brightness.")
        except Exception:
            pass  # Fallback to xbacklight or brightnessctl

        # Fallback tools
        if shutil.which("xbacklight"):
            os.system(f"xbacklight -inc {step} > /dev/null")
        elif shutil.which("brightnessctl"):
            os.system(f"brightnessctl set {step}%+ > /dev/null")
        else:
            print("No supported brightness tool found (xbacklight, brightnessctl, or sysfs).")

    elif system == "Windows":
        try:
            import wmi
            wmi_obj = wmi.WMI(namespace='wmi')
            methods = wmi_obj.WmiMonitorBrightnessMethods()[0]
            brightness = wmi_obj.WmiMonitorBrightness()[0].CurrentBrightness
            new_brightness = min(brightness + step, 100)
            methods.WmiSetBrightness(new_brightness, 0)
        except Exception as e:
            print(f"Failed to adjust brightness on Windows: {e}")

    elif system == "Darwin":  # macOS
        if shutil.which("brightness"):
            os.system(f"brightness -v +{step/100.0}")
        else:
            print("Install 'brightness' tool with: brew install brightness")
    else:
        print("Unsupported OS for brightness control.")


def decrease_brightness(step=10):
    system = platform.system()

    if system == "Linux":
        try:
            path = "/sys/class/backlight/intel_backlight"
            with open(f"{path}/brightness", "r") as f:
                current = int(f.read().strip())
            with open(f"{path}/max_brightness", "r") as f:
                max_val = int(f.read().strip())

            new = max(current - int((step / 100) * max_val), 1)
            with open(f"{path}/brightness", "w") as f:
                f.write(str(new))
            return
        except PermissionError:
            print("Run as root to change brightness.")
        except Exception:
            pass

        if shutil.which("xbacklight"):
            os.system(f"xbacklight -dec {step} > /dev/null")
        elif shutil.which("brightnessctl"):
            os.system(f"brightnessctl set {step}%- > /dev/null")
        else:
            print("No supported brightness tool found (xbacklight, brightnessctl, or sysfs).")

    elif system == "Windows":
        try:
            import wmi
            wmi_obj = wmi.WMI(namespace='wmi')
            methods = wmi_obj.WmiMonitorBrightnessMethods()[0]
            brightness = wmi_obj.WmiMonitorBrightness()[0].CurrentBrightness
            new_brightness = max(brightness - step, 1)
            methods.WmiSetBrightness(new_brightness, 0)
        except Exception as e:
            print(f"Failed to adjust brightness on Windows: {e}")

    elif system == "Darwin":  # macOS
        if shutil.which("brightness"):
            os.system(f"brightness -v -{step/100.0}")
        else:
            print("Install 'brightness' tool with: brew install brightness")
    else:
        print("Unsupported OS for brightness control.")

def scroll_based_on_motion(landmarks):
    global prev_finger_positions

    # Current y positions
    index_y = landmarks[8].y
    middle_y = landmarks[12].y
    ring_y = landmarks[16].y

    if all(prev_finger_positions.values()): # Ensure previous positions are initialized
        # Check if fingers are moving in the same direction
        if (
            index_y < prev_finger_positions["index"] and
            middle_y < prev_finger_positions["middle"] and
            ring_y < prev_finger_positions["ring"]
        ):
            pyautogui.scroll(SCROLL_SENSITIVITY)  # Scroll Up
        elif (
            index_y > prev_finger_positions["index"] and
            middle_y > prev_finger_positions["middle"] and
            ring_y > prev_finger_positions["ring"]
        ):
            pyautogui.scroll(-SCROLL_SENSITIVITY)  # Scroll Down

    # Update previous positions
    prev_finger_positions["index"] = index_y
    prev_finger_positions["middle"] = middle_y
    prev_finger_positions["ring"] = ring_y

    time.sleep(0.1)

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?).")
        break

    frame_counter += 1
    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[idx].classification[0].label
            is_left = label == "Left"
            is_right = label == "Right"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)
            ring_x, ring_y = int(landmarks[16].x * w), int(landmarks[16].y * h)
            pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)
            wrist_x = int(landmarks[0].x * w)

            cv.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)

            if is_right and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:
                mouse_x = int(np.clip(index_x * (screen_width / w), 0, screen_width - 1))
                mouse_y = int(np.clip(index_y * (screen_height / h), 0, screen_height - 1))
                mouse.position = (mouse_x, mouse_y)


            if is_right and np.hypot(index_x - thumb_x, index_y - thumb_y) < CLICK_THRESHOLD:
                mouse.click(Button.left, 1)
                time.sleep(0.2)

            if is_right and np.hypot(middle_x - thumb_x, middle_y - thumb_y) < CLICK_THRESHOLD:
                mouse.click(Button.right, 1)
                time.sleep(0.2)

            if is_right and prev_index_x is not None and prev_index_y is not None and frame_counter % 3 == 0:
                movement = np.hypot(index_x - prev_index_x, index_y - prev_index_y)
                if movement > 5:
                    mouse.scroll(0, -1 if index_y < h // 2 else 1)
                    time.sleep(0.2)

            if is_right:
                scroll_based_on_motion(landmarks)
            prev_index_x, prev_index_y = index_x, index_y

            if is_left and np.hypot(index_x - middle_x, index_y - middle_y) < FINGER_CLOSE_THRESHOLD:
                if control_mode == "volume":
                    if index_y < h // 2:
                        increase_volume()
                    else:
                        decrease_volume()
                else:
                    if index_y < h // 2:
                        increase_brightness()
                    else:
                        decrease_brightness()
                time.sleep(0.2)


            if is_left and np.hypot(pinky_x - thumb_x, pinky_y - thumb_y) < CLICK_THRESHOLD and time.time() - last_switch_time > CONTROL_SWITCH_COOLDOWN:
                control_mode = "brightness" if control_mode == "volume" else "volume"
                print(f"Switched to {control_mode} mode")
                last_switch_time = time.time()
                time.sleep(0.3)

            wave_positions.append(wrist_x)
            if len(wave_positions) > 10:
                wave_positions.pop(0)

            if len(wave_positions) >= 5:
                move_range = max(wave_positions) - min(wave_positions)
                if move_range > WAVE_THRESHOLD and time.time() - last_wave_time > MUTE_COOLDOWN:
                    toggle_mute()
                    print("Wave gesture detected: Toggled mute")
                    last_wave_time = time.time()
                    wave_positions.clear()

    frame = cv.addWeighted(frame, 1 - BLACKBOARD_OVERLAY_WEIGHT, blackboard, BLACKBOARD_OVERLAY_WEIGHT, 0)
    cv.imshow("Hand Gesture Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
