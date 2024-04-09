import cv2
import boto3
import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO

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
    Preprocess the frame for ONNX model.
    """
    # Resize and normalization steps here. Adjust according to your model's requirements.
    # Example for 640x640 input size and normalization.
    img = cv2.resize(frame, (640, 640))
    img = img / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)  # Change data layout from HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def segment_and_upload(source, bucket_name, video_key):
    # Load ONNX model
    ort_session = ort.InferenceSession('./best.onnx')

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
        input_blob = preprocess(frame)

        # Perform inference
        outputs = ort_session.run(None, {'images': input_blob}) # Adjust input name if necessary
        for r in outputs:
            print(r)
        # Post-processing steps here...
        # For demonstration, we just write the original frame
        if out is None:
            height, width, layers = frame.shape
            out = cv2.VideoWriter('result-onnx.mp4', fourcc, 20.0, (width, height))

        out.write(frame)  # Write the frame or the processed frame
        cap.release()
        out.release()



    # Upload the result video to S3
    # upload_to_s3(bucket_name, video_key, 'result-onnx.mp4')

# Example usage
segment_and_upload('video.mov', 'your_bucket_name', 'segmented_video.mp4')
