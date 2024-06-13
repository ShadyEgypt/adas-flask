import os
os.system("pip install ultralytics")
from ultralytics import YOLO
from PIL import Image
import cv2
import json


def resize(mask, height, width):
    # Resize the array to match the height of the target shape (1280, 720)
    resized_array = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_array

def load_model(model_name):
    compiled_model = YOLO(f'./{model_name}')
    H, W = 720, 1280
    return compiled_model, H, W

def preprocess(frame, H, W):
    """
    Preprocess the frame for yolov8 model.
    """
    image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(image_bgr, (W, H))
    return resized_image

def postprocess(results):
    """
    Postprocess the frame for yolov8 model.
    """
    for r in results:
        im_array = r.plot(probs=False,conf=False, boxes=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    return im

def segment(img_file_path, model_name, alpha = 0.3):
    try:
        compiled_model, H, W = load_model(model_name)
        img = cv2.imread(img_file_path)
        
        # Preprocess the frame
        height, width, _ = img.shape
        input_image = preprocess(img, H, W)

        # Perform inference
        results = compiled_model(input_image)

        # Post-processing steps here...
        output_image = postprocess(results)
        output_image.save(f'./result.png')

        body = {
            "message": "Image segmented!",
        }

        response = {
            "statusCode": 200,
            "body": json.dumps(body)
        }

        return response

    except Exception as e:
        error_message = f"Error : {str(e)}"
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": error_message})
        }
        return response


# response = segment('./f6156832-83248456.jpg')
# print(response)
