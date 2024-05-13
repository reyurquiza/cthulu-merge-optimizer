# CS-483 Final Project: 
## Real Time Game Decision-Making with an Object-Detection Model
-----
## Authors: 
- Rey Urquiza 
- Amelia Rotondo 

-----
## ToC: 
1. [About](#about)
2. [Deployment Instructions](#deployment-instructions)
3. [Steps to Run](#steps-to-run)

-----
## About:

This machine learning implementation project looks into the usage of a convolutional neural network, specifically YOLOv8, in a real-time gaming environment to improve decision-making processes. The purpose was to employ YOLOv8 for dynamic object detection in the game Cthulhu Merge, which involved processing visual input to influence game strategy. The application takes screenshots at regular intervals, uses the CNN to detect and classify game elements, then follows a predetermined logic to make the best move. Our findings indicate that employing YOLOv8 significantly improves the accuracy and speed of object recognition in the gaming environment, allowing for more strategic and responsive gameplay. This successfully tackles concerns with real-time data processing and game interaction, resulting in significant performance improvements.

## Deployment Instructions: 

### Developing a New Model
1. Run `yolo_model.py` to develop a model. 
2. In `main.py`, Replace the existing AI `model_path` with the newly-developed model.

### Setting-Up the Game Window 
1. Visit the [Cthulhu Merge](https://aplovestudio.itch.io/cthulhu-merge) Game Website
2. Click "Play Game" 
3. Ensure that the Entire Game Window is Visible 
4. You may want to reset the game-board by pressing X 

## Steps to Run:
1. In any terminal, run the provided `main.py` file.
2. You will be prompted to Calibrate the corners of the Game Window:
    - Before clicking, make sure not to include the right-hand heirarchy image as part of the game window, as that may cause bugs. 
        - First, click once on the Upper-Left Corner.
        - Second, click once on the Lower-Right Corner.
3. The AI will take a screenshot of the Game Window 
4. Let the AI play the game as long as you want. 
    - If it loses, it will automatically play another round. 
5. Close the Working Terminal to Terminate the Program.
