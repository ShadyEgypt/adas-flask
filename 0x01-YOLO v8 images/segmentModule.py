from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2
import os
import json

model = YOLO(f'./best.pt')


def segment(img_file_path):
    try:
        img = cv2.imread(img_file_path)
        results = model(img)

        for r in results:
            im_array = r.plot(probs=False,conf=False, boxes=False)  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save('result.jpg')

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
