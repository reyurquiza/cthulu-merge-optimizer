from ultralytics import YOLO
# from roboflow import Roboflow

if __name__ == "__main__":  # this is crucial
    model = YOLO('runs/detect/train12/weights/best.pt') # yolov8n.pt

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    # results = model.train(data='data.yaml', imgsz=640, epochs=100, workers=1)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model(['cthulu_dataset/test/images/test1.jpg', 'cthulu_dataset/test/images/test2.jpg', 'cthulu_dataset/test/images/test3.jpg'])  # return a list of Results objects

    print("Displaying results:\n")
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk

    # Export the model to ONNX format
    # success = model.export(format='onnx')


