import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np


# Load the MiDaS model version small
midasModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# Load the appropriate transforms for midas
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform 

# put the model in cpu or gpu and 
device = torch.device("cpu")  
midasModel.to(device)

# make the model ready for inference and not training
midasModel.eval()

# # Load YOLO model (yolov5n is a lightweight model)
#yoloModel = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')  


# Create FastAPI app
myapp = FastAPI()


@myapp.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the image bytes from the client
            image_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert img to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # prepare the img via transform and put it cpu so it can be ready for midas process
            midas_input_img = transform(imgRGB).to(device)         
            
            # maintain the non-gradient context when calling the prediciton (cause we in inference)
            with torch.no_grad():          
            
                #getting the prediction
                midasTensor = midasModel(midas_input_img)          
           
                #Resize the prediction to match the input image size before transforms
                midasTensor = torch.nn.functional.interpolate(
                midasTensor.unsqueeze(1),
                size=imgRGB.shape[:2],
                mode="bicubic",
                align_corners=False,
                ).squeeze()
        
            #Convert the prediction to a NumPy array cause it's easy to handle better than a pytorch tensor
            midasNumpy = midasTensor.cpu().numpy()

            #apply yolo inference
            # results = yoloModel(img)
            
            # # render the img inside the yolo results
            # results.render()  

            # # make a copy of it to use it
            # rendered_img = results.ims[0].copy()
            
            # #loop on detected objcts and for each one get the className, and get the meandepth of bbox then print it
            # for pred in results.xyxy[0]:      
            #     x1, y1, x2, y2, confidence, class_id = pred
                   
            #     depth_values = midasNumpy[int(y1):int(y2)+1, int(x1):int(x2)+1]
            #     mean_depth = np.nanmean(depth_values)        
            #     mean_depth_str = f"{mean_depth:.2f}"
        
            #     # Add the mean_depth text at the top of the bounding box
            #     cv2.putText(rendered_img, mean_depth_str, (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


            # Normalize the depth map to 0-255
            # Ensure that the depth map is scaled correctly
            midasNumpy = (midasNumpy - np.nanmin(midasNumpy)) / (np.nanmax(midasNumpy) - np.nanmin(midasNumpy)) * 255.0
            midasNumpy = np.clip(midasNumpy, 0, 255).astype(np.uint8)  # Convert to uint8


            # turn the numpy img to bytes
            img_buffer = BytesIO()
            Image.fromarray(midasNumpy).save(img_buffer, format='JPEG')
            rendered_image_bytes = img_buffer.getvalue()

            # send image bytes back via websocket
            await websocket.send_bytes(rendered_image_bytes)
           

            # Filter results by classes of interest
            # detected_objects = []
            # detected_objects_names = []
            # for pred in results.pred[0]:  # Loop over detections
            #     class_id = int(pred[-1])  # The class ID is the last value
            #     if class_id in classes_of_interest:
            #         detected_objects.append({
            #             'class': model.names[class_id],
            #             'confidence': float(pred[4]),  # Confidence score
            #             'bbox': [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]  # Bounding box
            #         })
            #         detected_objects_names.append(model.names[class_id])

            # # Send the detection results back to the client as JSON
            # await websocket.send_json({'detections': detected_objects_names})

    except Exception as e:
        await websocket.send_text(e)
        print(f"Error: {e}")
 






# def main():

    # # Load the MiDaS model version small
    # midasModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    # # Load the appropriate transforms for midas
    # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    # transform = midas_transforms.small_transform 
 
    
    # #get the image and turn to RGB so it can work with midas
    # img = cv2.imread("pics\ll.JPG")
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # height, width = img.shape[:2]

    # #put the model in cpu or gpu and make it ready for inference and not training
    # device = torch.device("cpu")  
    # midasModel.to(device)
    # midasModel.eval()

    # #prepare the img via transform and put it cpuXgpu so it can be ready for midas process
    # midas_input_img = transform(imgRGB).to(device)

    # #maintain the non-gradient context when calling the prediciton (cause we in inference)
    # with torch.no_grad():

    #     #getting the prediction
    #     midasTensor = midasModel(midas_input_img)

    #     # Resize the prediction to match the input image size before transforms
    #     midasTensor = torch.nn.functional.interpolate(
    #         midasTensor.unsqueeze(1),
    #         size=imgRGB.shape[:2],
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze()
 
    # # Convert the prediction to a NumPy array cause it's easy to handle better than a pytorch tensor
    # midasNumpy = midasTensor.cpu().numpy()



    #     ################################################################################


    # #Load YOLO model (yolov5n is a lightweight model)
    # yoloModel = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')  

    # #apply inference
    # results = yoloModel(img)
    # results.render()  
    # rendered_img = results.ims[0].copy()
    # #loop on detected objcts and for each one get the className, and get the meandepth of bbox then print it
    # for pred in results.xyxy[0]:      
    #     x1, y1, x2, y2, confidence, class_id = pred
   
    #     #class_name = yoloModel.names[int(class_id)] 

    #     depth_values = midasNumpy[int(y1):int(y2)+1, int(x1):int(x2)+1]
    #     mean_depth = np.nanmean(depth_values)        
    #     mean_depth_str = f"{mean_depth:.2f}"

        

    #     # Add the mean_depth text at the top of the bounding box
    #     cv2.putText(rendered_img, mean_depth_str, (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    #     cv2.imwrite("ilyaaas.jpg", rendered_img)


    
       
# if __name__ == "__main__":
#     main()


# import torch
# from PIL import Image
# from io import BytesIO
# from ultralytics import YOLO
# from fastapi import FastAPI, WebSocket

# # Load YOLO model (yolov5n is a lightweight model)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')

# # Create FastAPI app
# myapp = FastAPI()

# # Classes of interest (change as needed)
# classes_of_interest = [0, 1, 3, 2, 5, 7, 9, 11, 15, 16, 17, 18, 19]

# def process_image(image_path):
#     # Load and process the image
#     results = model(image_path)
    
#     # Render the bounding boxes on the image
#     results.render()  
    
#     # Convert the rendered image to bytes
#     img_buffer = BytesIO()
#     Image.fromarray(results.ims[0]).save(img_buffer, format='JPEG')
#     rendered_image_bytes = img_buffer.getvalue()
    
#     return rendered_image_bytes

# def main():
#     # Example image path (change this to your actual image path)
#     image_path = "pics/bus.jpg"
    
#     # Process the image and get the rendered bytes
#     rendered_image_bytes = process_image(image_path)
    
#     # Print the rendered image bytes
#     print("Rendered image bytes length:", len(rendered_image_bytes))
    
#     # Optional: Save the rendered image to a file for verification
#     with open('rendered_image.jpg', 'wb') as f:
#         f.write(rendered_image_bytes)
#     print("Rendered image saved as 'rendered_image.jpg'")

# if __name__ == "__main__":
#     main()