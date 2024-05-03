import pandas as pd
import pytesseract  
import cv2
import pickle 
import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CSV_PATH = 'dataset/gameStates/scores.csv'

# Preprocessing Schema
def preprocess_img(image_path, display=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Check the file path.")
    
    # Resize the image to the desired dimensions
    # Crop to get only the top-left corner
    cropped_image = image[100:250, :350]
    resized_image = cv2.resize(cropped_image, (450, 350))

    # Convert images to a suitable format (e.g., flatten arrays, normalize)
    norm_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    norm_image = norm_image / 255.0  # Normalize pixel values
    ready_image = norm_image.flatten()
    
    if display:
        cv2.imshow('Preprocessed Image', norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return ready_image


# Reads and Preprocesses the Image
def read_and_preprocess(csv_path):
    # For Testing
    display = False
    
    # Get the current working directory
    cwd = os.getcwd()
    print("Current working directory:", cwd)

    # Load CSV file into DataFrame
    data = pd.read_csv(csv_path)
    preprocessed_images = []  # Initialize a list to store preprocessed images

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        image_name = row['image_name']
        score = row['score']

        # Construct the path to the image relative to the current working directory
        image_path = os.path.join(cwd, 'dataset', 'gameStates', image_name)
        image_path = image_path + ".png"  # Add the file extension
            
        # Preprocess the image
        try:
            processed_image = preprocess_img(image_path, display)
            preprocessed_images.append((processed_image, score))  # Append the processed image and its score to the list
            print(f"Processed Image {index+1}/{len(data)}: {image_path}")
            print(f"Processed Score {index+1}/{len(data)}: {score}\n")
        except Exception as e:
            print(f"Failed to process image at {image_path}: {e}")
    
    return preprocessed_images  # Return the list of preprocessed images and scores

# Normalizes the Data for Training
def prepare_data_for_training(preprocessed_data):

    X = []  # Image data
    y = []  # Labels (scores)
    for image, score in preprocessed_data:

        # Append flattened image to X
        X.append(image)
        # Append label to y
        y.append(score)

    # Convert lists to numpy arrays for machine learning processing
    X = np.array(X)
    y = np.array(y)

    return X, y

# Split the Dataset (Training and Testing)
def split_dataset(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the Model
def train_model(X_train, y_train):
    # Initialize and train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    # Predict the scores on the testing set
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"Actual Scores: {y_test}")
    # Calculate the mean squared error between the predicted and actual scores
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return mse

# Actually Produces a Model
def train_score_model(csv_path):
    try:
        # Preprocessing
        preprocessed_data = read_and_preprocess(csv_path)
        
        # Extracting Lists for Training
        X, y = prepare_data_for_training(preprocessed_data)
        
        # Train the Model to get 95% Accuracy (MSE)
        mse = float('inf')
        while mse > 0.05:  # Continue training until the MSE is less than or equal to 0.05
            X_train, X_test, y_train, y_test = split_dataset(X, y)
            model = train_model(X_train, y_train)
            mse = evaluate_model(model, X_test, y_test)
            print(f"Current MSE: {mse}")
        
        # Save the model with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'model_{timestamp}.sav'
        pickle.dump(model, open(filename, 'wb'))
        print(f"Model saved as {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Example usage
train_score_model(CSV_PATH)