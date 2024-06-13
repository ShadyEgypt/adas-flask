import cv2
import json, os
import numpy as np
import openvino as ov
import pandas as pd
import time

core = ov.Core()

video_path = os.path.join('output_video.mp4')
csv_path = os.path.join('output_csv.csv')

def resize(mask, height, width):
    # Resize the array to match the height of the target shape (1280, 720)
    resized_array = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_array

def load_model(model_dir):
    model = core.read_model(model=model_dir)
    compiled_model = core.compile_model(model=model, device_name='CPU')
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    N, C, H, W = input_layer_ir.shape
    return compiled_model, H, W, output_layer_ir

def preprocess(frame, H, W):
    """
    Preprocess the frame for openvino model.
    """
    image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(image_bgr, (W, H))
    # Reshape to the network input shape.
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )  
    return input_image

def postprocess(frame, model_result, height, width, alpha = 0.3):
    """
    Postprocess the frame for visualization.
    """
    # Constants
    colormap = np.array([[0, 0, 0], [119, 255, 51], [120, 183, 53], [86, 210, 245]])
    # Divide the mask into four equal parts along the first axis
    parts = np.split(model_result, 4, axis=1)
    # Select the road mask and curbs parts
    bg = parts[0].reshape((512,896,1))
    road = parts[1].reshape((512,896,1))
    marks = parts[2].reshape((512,896,1))
    curbs = parts[3].reshape((512,896,1))
    # Rescale the values to the range [0, 255]
    bg_white_mask = bg * [255, 255, 255]
    road_white_mask = road * [255, 255, 255]
    curbs_white_mask = curbs * [255, 255, 255]
    marks_white_mask = marks * [255, 255, 255]
    road_colored_mask = road * colormap[1]
    marks_colored_mask = marks * colormap[2]
    curbs_colored_mask = curbs * colormap[3]
    # Change type of np array to avoid errors later
    bg_white_mask = bg_white_mask.astype('uint8')
    road_colored_mask = road_colored_mask.astype('uint8')
    marks_colored_mask = marks_colored_mask.astype('uint8')
    curbs_colored_mask = curbs_colored_mask.astype('uint8')
    road_white_mask = road_white_mask.astype('uint8')
    marks_white_mask = marks_white_mask.astype('uint8')
    curbs_white_mask= curbs_white_mask.astype('uint8')
    # Resize to original image width and height
    bg_white_mask = resize(bg_white_mask, height, width)
    road_white_mask = resize(road_white_mask, height, width)
    marks_white_mask = resize(marks_white_mask, height, width)
    curbs_white_mask= resize(curbs_white_mask, height, width)
    road_colored_mask = resize(road_colored_mask, height, width)
    marks_colored_mask = resize(road_colored_mask, height, width)
    curbs_colored_mask = resize(curbs_colored_mask, height, width)
    # Image Arthimatic Operations
    colored_mask = road_colored_mask + curbs_colored_mask
    white_mask = road_white_mask + curbs_white_mask + marks_white_mask
    # Bitwise Operations
    subtracted_road=cv2.bitwise_and(frame, bg_white_mask,mask=None)
    segmented_road=cv2.bitwise_and(frame, white_mask,mask=None)
    # Overlay
    alpha = .7
    overlay = segmented_road.copy()
    segmented_image = cv2.addWeighted(colored_mask, alpha, overlay, 1-alpha, 0, overlay)
    final_result=cv2.bitwise_or(subtracted_road,segmented_image,mask=None)

    return final_result


def segment(source):
    try:
        print("loading model")
        compiled_model, H, W, output_layer_ir = load_model('./models/road-segmentation-adas-0001.xml')
        # Open the video
        print("reading input video")
        cap = cv2.VideoCapture(source)
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

            # Preprocess the frame
            print("Executing preprocessing")
            image_h, image_w, _ = frame.shape
            input_image = preprocess(frame, H, W)
            infer_start_time = time.time()
            # Perform inference
            print("Executing inference")
            result = compiled_model([input_image])[output_layer_ir]
            infer_end_time = time.time()
            inference_times.append(infer_end_time - infer_start_time)
            # Post-processing steps here...
            print("Executing postprocessing")
            output_image = postprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), result, image_h, image_w, alpha=.3)
            print("result returned to output_fn ...")
            if out is None:
                out = cv2.VideoWriter(f'output_video.mp4', fourcc, 20.0, (image_w, image_h))

            out.write(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            print(f'frame {i} written')
            i += 1
            if (i == 100):
                break

        cap.release()
        if out is not None:
            out.release()
        
        # Saving the inference times to a DataFrame and then to a CSV file
        df = pd.DataFrame(inference_times, columns=['Inference Time'])
        df.to_csv(csv_path, index=True)
        print("Inference times saved as a csv file")

        body = {
            "message": "Video segmented!",
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
# segment('./data/ccdfec32-b738be17.mp4')
# segment('./data/cce8ee4f-ed9bff06.mp4')
# segment('./data/cce96cee-e92d6e05.mp4')
# segment('./data/cce124cb-c135d3ca.mp4')
# segment('./data/cce124cb-eaa1877e.mp4')
# segment('./data/cce887ff-443c174c.mp4')
# segment('./data/ccea6dde-45f9979a.mp4')
# segment('./data/ccea6dde-e46f1791.mp4')
# segment('./data/ccece5b4-960bf7a9.mp4')
# segment('./data/ccece5b4-22291259.mp4')

# response = segment('./data/alex-footage.mp4')
# print(response)