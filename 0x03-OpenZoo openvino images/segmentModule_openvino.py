import cv2
import json
import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
from notebook_utils import segmentation_map_to_image

core = ov.Core()

model = core.read_model(model='./model/road-segmentation-adas-0001.xml')
compiled_model = core.compile_model(model=model, device_name='CPU')

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)
N, C, H, W = input_layer_ir.shape

# Define colormap, each color represents a class.
colormap = np.array([[68, 1, 84], [51, 255, 119], [53, 183, 120], [199, 216, 52]])

# Define the transparency of the segmentation mask on the photo.
alpha = 0.3


def preprocess(frame):
    """
    Preprocess the frame for openvino model.
    """
    resized_image = cv2.resize(frame, (W, H))
    # Reshape to the network input shape.
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )  
    return input_image

def postprocess(frame, model_result, image_h, image_w):
    """
    Postprocess the frame for visualization.
    """
    segmentation_mask = np.argmax(model_result, axis=1)
    
    mask = segmentation_map_to_image(segmentation_mask, colormap)  # Ensure this returns a 3-channel image
    resized_mask = cv2.resize(mask, (image_w, image_h))
    # Create an image with mask.
    image_with_mask = cv2.addWeighted(resized_mask, alpha, frame, 1 - alpha, 0)
    return image_with_mask


def segment(img_file_path):
    try:
        img = cv2.imread(img_file_path)
        # Preprocess the frame
        image_h, image_w, _ = img.shape
        input_image = preprocess(img)

        # Perform inference
        result = compiled_model([input_image])[output_layer_ir]

        # Post-processing steps here...
        output_image = postprocess(img, result, image_h, image_w)

        cv2.imwrite('./result.jpg', output_image)
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
