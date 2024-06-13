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

def load_model(model):
    model = core.read_model(model='./models/semantic-segmentation-adas-0001.xml')
    
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

def postprocess(frame, model_result, height, width, alpha = 1):
    """
    Postprocess the frame for visualization.
    """
    # Constants
    colormap = np.array([[128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102],
                        [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
                        [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220],
                        [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0],
                        [100, 80, 0], [230, 0, 0], [32, 11, 119], [255, 255, 255]])
    result_array = np.vstack(model_result)
    result_array = np.squeeze(result_array, axis=0)
    result_array = cv2.resize(result_array.astype('float32'), (width, height))
    colored_mask = colormap[result_array.astype('uint8')]
    colored_mask = colored_mask.astype('uint8')
    overlay = frame.copy()
    # Add the segmentation mask to the overlay with transparency
    final_result = cv2.addWeighted(colored_mask, alpha, overlay, 1-alpha, 0, overlay)

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