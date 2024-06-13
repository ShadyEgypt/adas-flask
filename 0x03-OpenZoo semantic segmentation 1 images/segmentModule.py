import cv2
import json
import numpy as np
import openvino as ov

core = ov.Core()
image_path = './result.jpg'

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
        compiled_model, H, W, output_layer_ir = load_model('./models/road-segmentation-adas-0001.xml')
        img = cv2.imread(source)
        
        # Preprocess the frame
        height, width, _ = img.shape
        input_image = preprocess(img, H, W)

        # Perform inference
        result = compiled_model([input_image])[output_layer_ir]

        # Post-processing steps here...
        output_image = postprocess(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), result, height, width, alpha=.3)
        # Save the modified image
        cv2.imwrite(image_path, cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))        
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


# print(segment('./data/d3f34243-a7166713.jpg', 'road-segmentation-adas-0001'))
# print(segment('./data/d3f34243-a7166713.jpg', 'semantic-segmentation-adas-0001'))