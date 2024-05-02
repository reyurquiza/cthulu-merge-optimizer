import pyautogui
import time
from ultralytics import YOLO
from detect import capture_screen

MODEL = YOLO('runs/detect/train12/weights/best.pt')

def act_on_detections(results_obj, upcoming_icon_region=[0, 0, 100, 100]):
    boxes = results_obj['boxes']
    names = results_obj['names']
    probs = results_obj['probs']

    upcoming_icon = None
    highest_prob = 0
    for box, name, prob in zip(boxes, names, probs):
        x1, y1, x2, y2 = box

        if upcoming_icon_region[0] <= x1 <= upcoming_icon_region[2] and upcoming_icon_region[1] <= y1 <= upcoming_icon_region[3]:
            if prob > highest_prob:
                upcoming_icon = name
                highest_prob = prob

        if upcoming_icon is None:
            print("No upcoming icon detected.")
            return
        
        print(f"Upcoming icon to match: {upcoming_icon}")

        for box, name in zip(boxes, names):
            if name == upcoming_icon:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                pyautogui.click(center_x, center_y)
                print(f"Clicked on {name} at [{center_x}, {center_y}]")
                break

def main():
    try:
        while True:
            # Step 1: Capture Screen
            image = capture_screen()

            # Step 2: Run image throught model
            results = MODEL(image) # boxes, scores, classes

            # Step 3: Act on results
            act_on_detections(results)

            # Depends for performance
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()