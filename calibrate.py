from pynput import mouse
import pyautogui
import os

def on_click(x, y, button, pressed):
    if pressed:
        coordinates.append((x, y))
        print(f"Mouse clicked at ({x}, {y})")
        if len(coordinates) == 2:  # Stops listener after two clicks
            return False

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
