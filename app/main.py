import torch
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import json

# Load yolo model (the nano version) and configure it to run on cpu
yoloModel = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')

# our fastapi app
myapp = FastAPI()


@myapp.websocket("/yolo")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # we will stay ready for any incomes from our client
        while True:

            # receive the image bytes from the client
            image_bytes = await websocket.receive_bytes()

            # convert bytes to a numpy array for yolo
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # apply yolo inference
            results = yoloModel(img)

            detected_objects = []
            # iterate over each prediction
            for pred in results.xyxy[0]:
                x1, y1, x2, y2, confidence, class_id = pred
                
                # filter out objects with confidence less than 0.35
                if confidence >= 0.35:
                    # get the class name from the class_id
                    class_name = yoloModel.names[int(class_id)]
                    
                    # add the object to the detected_objects list as a dictionary
                    detected_objects.append({
                        'class_name': class_name,
                        'coordinates': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': confidence.item()
                    })

            # convert the list to json
            detected_objects_json = json.dumps(detected_objects)

            # send the json data back to our client
            await websocket.send(detected_objects_json)


    except Exception as e:
        print(f"Error: {e}")
        try:
            # send back error msg in json format
            error_message = json.dumps({"error": str(e)})
            await websocket.send(error_message)
        except Exception as send_error:
            print(f"Error sending error message: {send_error}")
        finally:
            # close the connection
            await websocket.close()

