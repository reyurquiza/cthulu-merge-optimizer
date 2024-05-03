import pyautogui
import keras
import cv2
import numpy as np
from PIL import Image

MODEL = keras.saving.load_model('CNN_model.keras')


def process_screenshot(x1, y1, x2, y2):
    # Take a screenshot of the specified area
    # now = datetime.datetime.now()
    # time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    image = pyautogui.screenshot(region=(x1, y1, x2, y2))

    # image.save('screenshot_{}.png'.format(time_string))

    img = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    # Convert the image to a format that OpenCV can work with
    # PIL image to numpy array
    img_np = np.array(img)
    
    # Convert RGB to BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve the detection process
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply a binary threshold to the image
    # Adjust the threshold value and maxVal to better suit the visuals
    _, threshold_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)

    # OR, find contours on the thresholded image to identify individual objects
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on top of the original image for visualization
    contour_img = cv2.drawContours(img_np.copy(), contours, -1, (0, 255, 0), 3)

    # Display the original, grayscale, and thresholded images
    cv2.imshow('Original Image', img_np)
    cv2.imshow('Grayscale Image', gray_img)
    cv2.imshow('Threshold Image', threshold_img)
    cv2.imshow('Contours Image', contour_img)

    boxes, scores, classes = MODEL.predict(img_np)



process_screenshot(50, 50, 100, 100)

