import cv2
import boto3
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

def upload_to_s3(bucket_name, file_key, file_path):
    """
    Uploads a file to an S3 bucket.
    
    :param bucket_name: Name of the S3 bucket.
    :param file_key: S3 key that the file will be uploaded as.
    :param file_path: The path to the file to upload.
    """
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, file_key)

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


def segment_and_upload(source, bucket_name, video_key):
    i = 0
    # Open the video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video file")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        image_h, image_w, _ = frame.shape
        input_image = preprocess(frame)

        # Perform inference
        result = compiled_model([input_image])[output_layer_ir]

        # Post-processing steps here...
        output_image = postprocess(frame, result, image_h, image_w)
        # For demonstration, we just write the original frame
        if out is None:
            height, width, layers = frame.shape
            out = cv2.VideoWriter('result-openvino.mp4', fourcc, 20.0, (width, height))

        out.write(output_image)  # Write the frame or the processed frame
        print(f'frame {i} written', i)
        i+=1
    cap.release()
    out.release()



    # Upload the result video to S3
    # upload_to_s3(bucket_name, video_key, 'result-onnx.mp4')

# Example usage
segment_and_upload('./data/video.mov', 'your_bucket_name', 'segmented_video.mp4')
