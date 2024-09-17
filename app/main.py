import torch
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
import websockets
import time
from fastapi.responses import HTMLResponse





# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')

# img = Image.open('pics/lgdg.jpg')

# results = model(img)

# results.save('results')

# classes_of_interest = [0, 1, 3, 2, 5, 7, 9, 11, 15, 16, 17, 18, 19]

# detected_objects = []


# Access details of detected objects like bounding boxes, confidence, and labels
# for pred in results.pred:
#     for obj in pred:
#         bbox = obj[:4]  # Bounding box coordinates (x1, y1, x2, y2)
#         confidence = obj[4]  # Confidence score
#         class_id = int(obj[5])  # Class ID

#         # Filter only the classes of interest
#         if class_id in classes_of_interest:
#             class_name = results.names[class_id]  # Get the class name            
#             # Create a map (dictionary) for the detected object
#             detected_obj = {
#                 'class_name': class_name,
#                 'bbox': bbox.tolist(),  # Convert tensor to list
#                 'confidence': round(confidence.item(), 2),  # Convert to float and round off
#                 'class_id': class_id
#             }
            
#             # Add the detected object map to the list
#             detected_objects.append(detected_obj)



# @myapp.get("/")
# def returnResult():
#     return 'heey brother'






myapp = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@myapp.get("/")
async def get():
    return HTMLResponse(html)


@myapp.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text("abooooood")


