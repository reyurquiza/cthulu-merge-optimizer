import pandas as pd
import pytesseract  
import cv2
import os

CSV_PATH = 'dataset/gameStates/scores.csv'

# Get the Score using PyTesseract 
def get_score(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Check the file path.")
    
    # Crop to get only the top-left corner
    cropped_image = image[100:250, :350]
    resized_image = cv2.resize(cropped_image, (350, 150))

    # Debugging w/ PyTesseract Checking
    score = pytesseract.image_to_string(resized_image, config='--psm 9 outputbase digits')
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
        print(f"Image: {image_name}, Actual Score: {score}, Detected Score: {detected_score}")
