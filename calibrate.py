from pynput import mouse
import pyautogui
import os
import cv2
from detect import capture_screen
import datetime



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
    coordinates = []
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    listener.join()

    # Calculate the top-left and bottom-right coordinates
    x1, y1 = min(coordinates[0][0], coordinates[1][0]), min(coordinates[0][1], coordinates[1][1])
    x2, y2 = max(coordinates[0][0], coordinates[1][0]), max(coordinates[0][1], coordinates[1][1])

    # Calculate width and height of the rectangle
    width = x2 - x1
    height = y2 - y1

    # Create 'screenshots' folder if it does not exist
    folder_path = "screenshots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Take a screenshot of the selected area

    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    file_path = os.path.join(folder_path, "selected_area_screenshot.png")
    screenshot.save(file_path)

    print(f"Screenshot of the area [{x1}, {y1}, {x2}, {y2}] saved as '{file_path}'")

#### TESTING CODE: 
#capture()
#calibrate()

# Gets Two Clicks- Returns the Coordinates [x1, y1, width, height]
def calibrate():
    
    coordinates = []
    
    def on_click(x, y, button, pressed):
        if pressed:
            coordinates.append((x, y))
            print(f"Mouse clicked at ({x}, {y})")
            if len(coordinates) == 2:  # Stops listener after two clicks
                return False
        
    print('Calibrating... Click Corners of the Game Window')
    
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    listener.join()

    # Calculate the top-left and bottom-right coordinates
    # Rey - had to turn the coords to ints because math with floats didnt work
    x1, y1 = int(min(coordinates[0][0], coordinates[1][0])), int(min(coordinates[0][1], coordinates[1][1]))
    x2, y2 = int(max(coordinates[0][0], coordinates[1][0])), int(max(coordinates[0][1], coordinates[1][1]))

    # Calculate width and height of the rectangle
    width = x2 - x1
    height = y2 - y1

    return x1, y1, width, height