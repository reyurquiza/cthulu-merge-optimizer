from pynput import mouse
import pyautogui
import os
import cv2
from detect import capture_screen
import datetime

# This is SOOOO Lazy but I'm just debugging anyways :P 
coordinates = []

def on_click(x, y, button, pressed):
    if pressed:
        coordinates.append((x, y))
        print(f"Mouse clicked at ({x}, {y})")
        if len(coordinates) == 2:  # Stops listener after two clicks
            return False

# Trying out Capture_Screen
def capture():
    # Generate a timestamped filename for the screenshot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    
    # Capture the screen using the capture_screen function from the detect module
    screenshot = capture_screen()
    
    # Save the screenshot to a file
    cv2.imwrite(filename, screenshot)
    print(f"Screenshot saved as {filename}")

# Using PyAutoGUI All-Properly
def test_calibrate():
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    listener.join()

    # Calculate the top-left and bottom-right coordinates
    x1, y1 = min(coordinates[0][0], coordinates[1][0]), min(coordinates[0][1], coordinates[1][1])
    x2, y2 = max(coordinates[0][0], coordinates[1][0]), max(coordinates[0][1], coordinates[1][1])

    # Calculate width and height of the rectangle
    width = x2 - x1
    height = y2 - y1
    print(f"Width: {width}, Height: {height}")

    # Create 'screenshots' folder if it does not exist
    folder_path = "screenshots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Take a screenshot of the selected area

    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    file_path = os.path.join(folder_path, "selected_area_screenshot.png")
    screenshot.save(file_path)

    print(f"Screenshot of the area [{x1}, {y1}, {x2}, {y2}] saved as '{file_path}'")

# You Click Twice, and it Returns the x1, y1, x2, y2 as a List 
def calibrate():
    coords = []
    
    def calibrate_onclick(x, y, button, pressed):
        if pressed:
            coords.append((x, y))
            print(f"Mouse clicked at ({x}, {y})")
            if len(coords) == 2:  # Stops listener after two clicks
                return False
    
    listener = mouse.Listener(on_click=calibrate_onclick)
    listener.start()
    listener.join()
    
    x1, y1 = min(coords[0][0], coords[1][0]), min(coords[0][1], coords[1][1])   
    x2, y2 = max(coords[0][0], coords[1][0]), max(coords[0][1], coords[1][1])   
    
    width = x2 - x1
    height = y2 - y1
    
    return [x1, y1, width, height]

#### TESTING CODE: 
#capture()
#test_calibrate()
#print(calibrate())
