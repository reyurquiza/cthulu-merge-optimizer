import pandas as pd
import pytesseract  
import cv2
import os
import numpy as np

CSV_PATH = 'dataset/gameStates/scores.csv'

# Get the Score using PyTesseract 
def get_score(path):
    # Load the Image
    img = cv2.imread(path)
        
    if img is None:
        raise ValueError("Could not load the image. Check the file path.")
    
    # Crop to get only the top-left corner
    img = cv2.resize(img, (1280, 715))
    img = img[150:260, 50:350]

    #  Preprocessing: 
        # Grayscale -> Thresholding -> Sharpening -> Erosion -> Dilation -> Gaussian Blur
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    img = cv2.erode(img, np.ones((4, 4), np.uint8), iterations=1)
    img = cv2.dilate(img, np.ones((4, 4), np.uint8), iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
    
    # Display the Image (DEBUGGING PURPOSES)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Debugging w/ PyTesseract Checking
    score = pytesseract.image_to_string(img, config='--psm 6 --oem 3 outputbase digits')
    if score == '':
        return '0'
    return score

# Get Test Scores 
def get_test_scores(csv_path):
    data = pd.read_csv(csv_path)
    test_scores = []
    for index, row in data.iterrows():
        image_name = row['image_name']
        score = row['score']
        test_scores.append((image_name, score))
    return test_scores

# Compares Get Score and Test Scores
def compare_scores(csv_path):
    test_scores = get_test_scores(csv_path)
    for image_name, score in test_scores:
        image_path = os.path.join('dataset', 'gameStates', image_name + ".png")
        detected_score = get_score(image_path)
        
        #  Printing Comparisson
        if int(score) == int(detected_score):
            print(f"Image: {image_name}, Score: {int(score)}")
        else:
            print(f"Image: {image_name}, Actual Score: {score}, Detected Score: {detected_score}")

# Testing Code
print(get_score('screenshots/selected_area_screenshot.png'))
compare_scores(CSV_PATH)