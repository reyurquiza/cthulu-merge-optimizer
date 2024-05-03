import time
import pyautogui

# Yes, this works for some reason.
def autoclicker(interval):
    while True:
        pyautogui.click()
        time.sleep(interval)

if __name__ == "__main__":
    # Set the click interval to 1 second
    autoclicker(1)
