from ultralytics import YOLO
from PIL import Image
import cv2
import os, json, time
import numpy as np
import pandas as pd

video_path = os.path.join('output_video.mp4')
csv_path = os.path.join('output_csv.csv')

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
        # convert im-array to cv2 image
        im = Image.fromarray(im_array)
        # convert PIL image to cv2 image
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return im

def segment(source):
    try:
        print("loading the model ...")
        compiled_model, H, W = load_model("models/best.pt")
        print("opening the video ...")
        # Open the video
        cap = cv2.VideoCapture(source)
        # get width and height of video frames
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened():
            print("Error opening video file")
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        i = 0
        inference_times = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            infer_start_time = time.time()
            print("Executing inference")
            results = compiled_model(frame)  # Ensure compiled_model is defined and can process frames
            infer_end_time = time.time()
            inference_times.append(infer_end_time - infer_start_time)

            print("Executing postprocessing")
            output_image = postprocess(results)  # Ensure postprocess function is defined

            if out is None:
                # Ensure img_width and img_height are defined according to your video/frame dimensions
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (img_width, img_height))

            # Convert color from RGB to BGR, as OpenCV expects BGR
            out.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            
            i += 1
            if i == 100:  # Process only the first 100 frames
                break

        # Release everything if job is finished
        cap.release()
        if out is not None:
            out.release()

        # Saving the inference times to a DataFrame and then to a CSV file
        df = pd.DataFrame(inference_times, columns=['Inference Time'])
        df.to_csv(csv_path, index=True)
        print("Inference times saved as a csv file")
        body = {
            "message": "Video segmented!"
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

# Example usage
# response = segment('data/alex-footage.mp4')
# print(response)
    