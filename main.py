import pyautogui
import numpy as np
import time
from ultralytics import YOLO
from detect import capture_screen
import random
from calibrate import calibrate


MODEL = YOLO('runs/detect/train12/weights/best.pt')

icon_levels = {
    2: 'Bunny',
    6: 'Frog',
    7: 'Sal',
    5: 'Fox',
    4: 'Dog',
    1: 'Boar',
    0: 'Bear',
    3: 'Deer',
    8: 'Cthulu'
}

def dumb_clicker(x, y):
    pyautogui.moveTo(x, y)
    time.sleep(.1)
    pyautogui.dragTo(button='left')
    pyautogui.click(clicks=2)

# replace with with a reinforcement learning neural network
def act_on_detections(results_obj):
    if not results_obj[0].boxes:
        dumb_clicker(random.randint(1000, 1520), random.randint(671, 700))
        # pyautogui.moveTo(random.randint(1000, 1520), random.randint(671, 700))
        # time.sleep(.1)
        # pyautogui.dragTo(button='left')
        # pyautogui.click(clicks=2)
        
    boxes = []
    names = []
    probs = []

    for result in results_obj:
        boxes = result.boxes.numpy()  # Boxes object for bounding box outputs
        names = result.boxes.cls.numpy()  # Keypoints object for pose outputs
        probs = result.boxes.conf.numpy()  # Probs object for classification outputs
        break
        
    # print("\n\nBoxes:\n", boxes)
    # print("\n\nNames:\n", names)
    # print("\n\nProbs:\n", probs)

    upcoming_icon = None
    highest_prob = 0
    upcoming_box = None
    for box, name, prob in zip(boxes, names, probs):

        x1, y1, x2, y2 = box.xyxy[0]

        if x2 < 350:
            # print(f"{x2} < 350?")
            # This means its the upcoming icon so we dont want to click on it
            upcoming_icon = name
            upcoming_box = box.xyxy[0]
            print(f"Upcoming icon: {icon_levels.get(upcoming_icon)}")
            highest_prob = prob
            break


    if upcoming_icon is None:
        print("No upcoming icon detected.")
        return
        
    for box, name in zip(boxes, names):
        # Make sure not to click on the 'upcoming' box
        if (upcoming_box==box.xyxy[0]).all():
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        print("Checking: ", icon_levels.get(name))
        if name == upcoming_icon:
            print(f"{icon_levels.get(name)} has been found at [{x1, y1, x2, y2}]!")
            center_x = (x1 + x2 + 1300) / 2
            center_y = (y1 + y2 + 600) / 2

            dumb_clicker(center_x, center_y)
            # pyautogui.click(center_x, center_y)
            # pyautogui.click(clicks=2)
                
            print(f"Clicked on {icon_levels.get(name)} at [{center_x}, {center_y}]")
            return
    # If no matching shapes in current game state click random
    dumb_clicker(random.randint(1000, 1520), random.randint(671, 700))
    # pyautogui.click(random.randint(1000, 1520), random.randint(671, 700))
    # pyautogui.click(clicks=2)
    print("No matches found, Im feeling lucky!")


def extract_features(game_screen):
    results = MODEL(game_screen)
    features = []
    for result in results:
        level = icon_levels.get(result['type'], 0)
        # normal_x = normalize_position(result['x'], game_screen.width)
        # normal_y = normalize_position(result['y'], game_screen.height)
        # features.append([level, normal_x, normal_y])
    return features


def main():
    print("Calibrating... Click Corners Now")
    screen = calibrate()
    
    # Creates a Screenshot Method called Capture
    capture = lambda: pyautogui.screenshot(region=screen) 
    
    print("Going to start now...")
    # pyautogui.click(random.randint(1000, 1520), random.randint(671, 700))
    try:
        while True:
            # Step 1: Capture Screen
            image = capture()
            # cv2.imshow('Test', image)
            # cv2.waitKey(0)

            # Step 2: Run image throught model
            results = MODEL(image) # boxes, scores, classes

            for result in results:
                result.save(filename='result.jpg')  # save to disk

            # Step 3: Act on results
            act_on_detections(results)

            # Depends for performance
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()