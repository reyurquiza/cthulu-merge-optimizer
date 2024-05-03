import csv
import os

##### Changes the csv file to have Relative Paths

# Get the absolute path to the directory where the script is running
script_directory = os.path.dirname(os.path.abspath(__file__))

# Function to convert an absolute path to a relative path (just the file name in this case)
def path_to_image_name(full_path):
    return os.path.relpath(full_path, script_directory)

# Read the existing CSV data
with open('scores.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# Modify the first column of each row except for the header to be relative paths
for i, row in enumerate(data):
    if i == 0:
        continue  # Skip the header row
    row[0] = path_to_image_name(row[0])

# Write the updated data back to the CSV
with open('scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("scores.csv has been updated with relative paths to the images.")
