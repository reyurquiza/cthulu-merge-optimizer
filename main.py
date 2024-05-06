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

# Returns the icon that is exactly one higher of the passed in icon or None if not found
def find_higher_rank_icon(current_rank):
    # Finding the icon with the next higher rank
    for icon, rank in icon_values.items():
        if rank == current_rank + 1:
            return icon
    return None  # No higher rank icon found

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
                if y1 >= y2 and abs(x1-x2) < h1:
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
    # 2 works better for mac, 10 for other
    for idx in range (1,9):
        pyautogui.dragTo(button='left')
        # print(idx)

# DOES THE FOLLOWING: 
    # Checks for pieces that match the upcoming piece
    # Checks if those pieces have animals above them
    # If they don't, then click on them
    # Otherwise, click on a random piece
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
       
    upcoming_value = None
    upcoming_prob = 0
    upcoming_box = None

    # Debugging
    # for box, name in zip(boxes, names):
        # print(f"{icon_levels.get(name)}: {box}")
    
    # Finding the Upcoming Box 
    for box, name, prob in zip(boxes, names, probs):

        # Gets the x, y, w, h of the box
        coords = box
        
        # Figures out if the Box is Off Board
        off_board = int(coords[0]) < int(screen_width/8)
        
        # # Debugging
        # print("-------------------------------------")
        # print('Coords:', coords)
        # print('Name:', icon_levels.get(name))
        # print(f"{int(coords[0])} < {int(screen_width/8)}? {off_board}")
        
        # Finding the Minimum Item
        if off_board:
            
            # This means its the upcoming icon so we dont want to click on it
            upcoming_value = name
            upcoming_box = coords
            
            # Removes it from the Collection
            boxes.remove(box)
            names.remove(name)
            probs.remove(prob)
            
            # print(f"Upcoming icon: {icon_levels.get(upcoming_value)}")
            upcoming_prob = prob
            break
    
    # Debugging
    # print("Boxes:\n", boxes)
    # print("Names:\n", names)
    upcoming_string = icon_levels.get(upcoming_value)
    print(f'Upcoming Item is: {upcoming_string} ({upcoming_value})\n')
        
    # Our Best Option
    best_var = 0
    best_box = None
    best_name = None
        
    # Finding the Best Chain (Boolean Logic)
    ####### We May Want to Change This to Reinforcement Learning

    current_rank = icon_values.get(upcoming_value)
    if current_rank is not None:
        higher_rank_icon = find_higher_rank_icon(current_rank)

    # Finding the rank above upcoming
    if higher_rank_icon is not None:
        for higher_box, higher_name in zip(boxes, names):
            if higher_name == higher_rank_icon and is_available(higher_box, boxes):
                print(f"\n\n[{upcoming_string}]: The rank above [{icon_levels.get(upcoming_value)}], which is [{icon_levels.get(higher_name)}], is available at [{higher_box}]]")

                # Calculate pos for dropping
                higher_center_x = higher_box[0] - higher_box[2]/4 + cs_diff[0]
                higher_center_y = higher_box[1] - higher_box[3]/4 + cs_diff[1]

                dumb_clicker(higher_center_x, higher_center_y)

                print(f"[{upcoming_string}]: Dropped [{icon_levels.get(upcoming_value)}] on [{icon_levels.get(higher_name)}] at [{higher_center_x}, {higher_center_y}]")
                return
                # end higher check
    print(f"[{upcoming_string}]: [{icon_levels.get(higher_rank_icon)}] is not available, will try to find a match instead.\n\n")

    # Finding the Same Animal
    for box, name in zip(boxes, names):
        # box = flatten(box)
        if name == upcoming_value:
            # If there is Anything Above the Animal
            if is_available(box, boxes):
                print(f"\n\n[{upcoming_string}]: {icon_levels.get(name)} has been found at [{box}]!")
    
                # Ratio Between Screenshot and Screen Size
                center_x = box[0] - box[2]/4 + cs_diff[0]
                center_y = box[1] - box[3]/4 + cs_diff[1]

                dumb_clicker(center_x, center_y)
                    
                print(f"[{upcoming_string}]: Clicked on {icon_levels.get(name)} at [{center_x}, {center_y}]\n\n")
                return
            else: 
                print(f'[{upcoming_string}]: Not Available!')  
        
    # If We Find Nothing? We Guess
    print("\n\n~~No matches found, Im feeling lucky!~~\n\n")
    dumb_clicker(random.randint(int(screen_width/3), int(2*screen_width/3)), random.randint(671, 700))
    return
    
# The Whole Thing!
def main():
    print("Going to start now...")
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