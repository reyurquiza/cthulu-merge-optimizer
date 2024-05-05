import pyautogui
import cv2 as cv2
import numpy as np
import time
from ultralytics import YOLO
from detect import capture_screen
import random
from pynput.mouse import Button, Controller
from calibrate import calibrate

# Computer Setup
mouse = Controller()
screen_width, screen_height = pyautogui.size()   

# AI Model Setup
MODEL = YOLO('runs/detect/train12/weights/best.pt')

# Deriving Names of Animals
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

# Values from Icons 
icon_values = {
    2: 0, # Bunny
    6: 1, # Frog
    7: 2, # Sal
    5: 3, # Fox
    4: 4, # Dog
    1: 5, # Boar
    0: 6, # Bear
    3: 7, # Deer
    8: 8  # Cthulhu
}

# Un-Nest a List 
flatten = lambda lst: [item for sublist in lst for item in sublist]

# Floats to Int
to_ints = lambda items: [to_ints(item) if isinstance(item, list) else int(item) for item in items]

# Returns True of Two Coords [x,y,w,h] Overlap 
def do_rectangles_overlap(rect1, rect2):
    # Extract the position and dimensions for both rectangles
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate the right and bottom coordinates of both rectangles
    right1, bottom1 = x1 + w1, y1 + h1
    right2, bottom2 = x2 + w2, y2 + h2

    # Check if there is overlap
    # Overlap happens if the right edge of one rectangle is farther to the right than the left edge of the other
    # and the bottom edge of one rectangle is farther down than the top edge of the other
    if (x1 < right2 and right1 > x2 and y1 < bottom2 and bottom1 > y2):
        return True
    else:
        return False

# Returns True of Two Coords [x,y,w,h], One is Above the Other
def is_available(rect1, list1):
    x1, y1, w1, h1 = rect1
    
    for rect2 in list1:
            if rect2 != rect1:
                # Extract the position and dimensions for both rectangles
                x2, y2, w2, h2 = rect2
                
                # Debugging
                # print(f'does {rect2} overlap {rect1}?')
                    
                # Check if the bottom edge of rect1 is above the top edge of rect2
                if (y1-h1/2) <= (y2-h2/2) and abs((x2-w2/2)-(x1-w1/2)) < h1:
                    return False
        
    return True

# Used for Removing the Minimum Tuple from Some Lists
def remove_min_x_element(z, x_list, y_list, w_list):
    # Find the index of the element with the smallest x value
    min_index = min(range(len(x_list)), key=lambda i: x_list[i])

    # Remove and return the elements at this index from all lists
    x_value = x_list.pop(min_index)
    y_value = y_list.pop(min_index)
    w_value = w_list.pop(min_index)
    
    return (x_value, y_value, w_value)

# Clicks the Screen at Coords x, y
def dumb_clicker(x,y):
    pyautogui.moveTo(x, y)
    time.sleep(.3)
    for idx in range (1,10):
        pyautogui.dragTo(button='left')
        # print(idx)

# replace with with a reinforcement learning neural network
def act_on_detections(results_obj, cs_diff):
    
    if not results_obj[0].boxes:
        dumb_clicker(random.randint(int(screen_width/3), int(2*screen_width/3)), random.randint(671, 700))

    boxes = []
    names = []
    probs = []

    for result in results_obj:
        boxes = to_ints((result.boxes.xywh.numpy()).tolist())  # Boxes object for bounding box outputs
        names= to_ints((result.boxes.cls.numpy()).tolist()[:])  # Keypoints object for pose outputs
        probs = to_ints((result.boxes.conf.numpy()).tolist())  # Probs object for classification outputs
       
    upcoming_icon = None
    highest_prob = 0
    upcoming_box = None
    
    # Finding the Upcoming Box 
    for box, name, prob in zip(boxes, names, probs):

        # Gets the x, y, w, h of the box
        coords = box
        
        # Figures out if the Box is Off Board
        off_board = int(coords[0]) < int(screen_width/4)
        
        # # Debugging
        # print('Coords:', coords)
        # print('Name:', icon_levels.get(name))
        # print(f"{int(coords[0])} < {int(screen_width/4)}? {off_board}")
        
        # Finding the Minimum Item
        if off_board:
            
            # This means its the upcoming icon so we dont want to click on it
            upcoming_icon = name
            upcoming_box = coords
            
            # Removes it from the Collection
            boxes.remove(box)
            names.remove(name)
            probs.remove(prob)
            
            # print(f"Upcoming icon: {icon_levels.get(upcoming_icon)}")
            highest_prob = prob
            
            break
    
    # Debugging
    # print("Boxes:\n", boxes)
    # print("Names:\n", names)
    print(f'Upcoming Item is: {icon_levels.get(upcoming_icon)} ({upcoming_icon})\n')
        
    # Our Best Option
    best_var = 0
    best_box = None
    best_name = None
        
    # Finding the Best Chain (Boolean Logic)
    ####### We May Want to Change This to Reinforcement Learning

    # Finding the Same Animal
    for box, name in zip(boxes, names):
        
        # box = flatten(box)
        if name == upcoming_icon:
            
            # If there is Anything Above the Animal
            if is_available(box, boxes):
                print(f"{icon_levels.get(name)} has been found at [{box}]!")
    
                # Ratio Between Screenshot and Screen Size
                center_x = box[0] - box[2]/4 + cs_diff[0]
                center_y = box[1] - box[3]/4 + cs_diff[1]

                dumb_clicker(center_x, center_y)
                    
                print(f"Clicked on {icon_levels.get(name)} at [{center_x}, {center_y}]\n\n")
                return
            else: 
                print('Not Available-')   
        
    # If We Find Nothing? We Guess 
    dumb_clicker(random.randint(int(screen_width/3), int(2*screen_width/3)), random.randint(671, 700))
    print("\n\n~~No matches found, Im feeling lucky!~~\n\n")
    
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
    print("Going to start now...")
    # pyautogui.click(random.randint(1000, 1520), random.randint(671, 700))
    screen = calibrate()
    
    # Capture-Screen Ratio
    cs_diff = [screen[0], screen[1]]
    time.sleep(1)    
        
    # # Debugging
    # print(screen_width, screen_height)
    # print(screen)
    # print('CS-Ratio: ',cs_diff)

    try:
        while True:
            # Step 1: Capture Screen
            image = pyautogui.screenshot(region=screen)

            # cv2.imshow('Test', image)
            # cv2.waitKey(0)

            # Step 2: Run image throught model
            results = MODEL(image) # boxes, scores, classes

            for result in results:
                result.save(filename='result.jpg')  # save to disk

            # Step 3: Act on results
            act_on_detections(results, cs_diff)

            # Depends for performance
            time.sleep(3)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()