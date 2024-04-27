import cv2
import json
import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
from notebook_utils import segmentation_map_to_image

core = ov.Core()

def resize(mask, height, width):
    # Resize the array to match the height of the target shape (1280, 720)
    resized_array = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_array

def load_model(model):
    if(model=='pedestrian-and-vehicle-detector-adas-0001'):
        model = core.read_model(model='./models/pedestrian-and-vehicle-detector-adas-0001.xml')
    elif(model=='person-vehicle-bike-detection-crossroad-0078'):
        model = core.read_model(model='./models/person-vehicle-bike-detection-crossroad-0078.xml')
    else:
        print('no model found!')
        return 0
    
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

def postprocess(frame, model_name, model_result, height, width, alpha = 0.3):
    """
    Postprocess the frame for visualization.
    """
    colormap = [(68, 1, 84), (51, 255, 119), (53, 183, 120), (199, 216, 52)]
    if(model_name=='pedestrian-and-vehicle-detector-adas-0001'):
        for result in model_result[0][0]:
            score = result[2]
            if (score>.70):
                print(score)
                x1, y1, x2, y2 = int(result[3] * width), int(result[4] * height), int(result[5] * width), int(result[6] * height)
                print(x1, y1, x2, y1)
                det_label = 'vehicle' if result[1]==1 else 'pedestrian'
                color =  colormap[0] if result[1]==1 else colormap[1]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, '{} {:.1%}'.format(det_label, score),
                            (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, .5, color, 1)
    elif(model_name=='person-vehicle-bike-detection-crossroad-0078'):
        for result in model_result[0][0]:
            score = result[2]
            if (score>.70):
                print(score)
                x1, y1, x2, y2 = int(result[3] * width), int(result[4] * height), int(result[5] * width), int(result[6] * height)
                print(x1, y1, x2, y2)
                if(result[1]==0):
                    det_label = 'vehicle'
                    color =  colormap[0]
                elif(result[1]==1):
                    det_label = 'pedestrian'
                    color =  colormap[1]
                else:
                    det_label = 'bike'
                    color =  colormap[2]
                
                print(det_label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, '{} {:.1%}'.format(det_label, score),
                            (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, .5, color, 1)
    return frame

def segment(img_file_path, model_name, alpha = 0.3):
    try:
        compiled_model, H, W, output_layer_ir = load_model(model_name)
        img = cv2.imread(img_file_path)
        
        # Preprocess the frame
        height, width, _ = img.shape
        input_image = preprocess(img, H, W)

        # Perform inference
        result = compiled_model([input_image])[output_layer_ir]

        # Post-processing steps here...
        output_image = postprocess(img, model_name, result, height, width, alpha)

        # Save the modified image
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


# segment('./data/test2.jpg', 'pedestrian-and-vehicle-detector-adas-0001')