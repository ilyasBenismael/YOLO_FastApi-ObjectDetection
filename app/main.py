import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np


# Load YOLO model (yolov5n is a lightweight model)
yoloModel = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')  

# Create FastAPI app
myapp = FastAPI()

#

@myapp.websocket("/yolo")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the image bytes from the client
            image_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # apply yolo inference
            results = yoloModel(img)
            
            # render the img inside the yolo results
            results.render()    

            # make a copy of it to use it
            rendered_img = results.ims[0].copy()
            
            # turn the numpy img to bytes
            img_buffer = BytesIO()
            Image.fromarray(rendered_img).save(img_buffer, format='JPEG')
            rendered_image_bytes = img_buffer.getvalue()

            # send image bytes back via websocket
            await websocket.send_bytes(rendered_image_bytes)
           

    except Exception as e:
        await websocket.send_text(e)
        print(f"Error: {e}")













            # #loop on detected objcts and for each one get the className, and get the meandepth of bbox then print it
            # for pred in results.xyxy[0]:      
            #     x1, y1, x2, y2, confidence, class_id = pred
                   
            #     depth_values = midasNumpy[int(y1):int(y2)+1, int(x1):int(x2)+1]
            #     mean_depth = np.nanmean(depth_values)        
            #     mean_depth_str = f"{mean_depth:.2f}"
        
            #     # Add the mean_depth text at the top of the bounding box
            #     cv2.putText(rendered_img, mean_depth_str, (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
