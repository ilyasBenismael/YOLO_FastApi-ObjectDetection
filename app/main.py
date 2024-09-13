import torch
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI






myapp = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

img = Image.open('pics/lgdg.jpg')

results = model(img)

results.save('results')

classes_of_interest = [0, 1, 3, 2, 5, 7, 9, 11, 15, 16, 17, 18, 19]

detected_objects = []


# Access details of detected objects like bounding boxes, confidence, and labels
for pred in results.pred:
    for obj in pred:
        bbox = obj[:4]  # Bounding box coordinates (x1, y1, x2, y2)
        confidence = obj[4]  # Confidence score
        class_id = int(obj[5])  # Class ID

        # Filter only the classes of interest
        if class_id in classes_of_interest:
            class_name = results.names[class_id]  # Get the class name            
            # Create a map (dictionary) for the detected object
            detected_obj = {
                'class_name': class_name,
                'bbox': bbox.tolist(),  # Convert tensor to list
                'confidence': round(confidence.item(), 2),  # Convert to float and round off
                'class_id': class_id
            }
            
            # Add the detected object map to the list
            detected_objects.append(detected_obj)



@myapp.get("/")
def returnResult():
    return {'results':detected_objects}
    
           

