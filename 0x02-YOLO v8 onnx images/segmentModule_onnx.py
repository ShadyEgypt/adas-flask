import cv2
import json
from yoloseg import YOLOSeg

model_path = "model/best.onnx"
model = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)



def segment(img_file_path):
    try:
        img = cv2.imread(img_file_path)
        boxes, scores, class_ids, masks = model(img)

        output_image = model.draw_masks(img)
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
