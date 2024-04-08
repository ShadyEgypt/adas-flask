import boto3
import cv2
import json
from io import BytesIO


def upload_to_s3(image, bucket_name, file_key):
    s3 = boto3.client('s3')
    _, buffer = cv2.imencode('.png', image)
    s3.upload_fileobj(BytesIO(buffer), bucket_name, file_key)

def overlay(image_name, img_file_path, mask_file_path):
    try:

        image = cv2.imread(img_file_path)
        mask = cv2.imread(mask_file_path)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        alpha = 0.4

        overlay = image.copy()

        # Add the segmentation mask to the overlay with transparency
        cv2.addWeighted(mask, alpha, overlay, 1 - alpha, 0, overlay)

        # Upload the resulting image to S3
        s3_bucket_name = 'adas-project-bucket'
        s3_file_key = f'results/{image_name}.jpg'
        upload_to_s3(overlay, s3_bucket_name, s3_file_key)
        output_url = f'https://ddx0brhffx34i.cloudfront.net/{s3_file_key}'
        body = {
            "message": "Overlay applied successfully!",
            "data": {
                "name": '{image_name}.jpg',
                "url": output_url
            },
        }

        response = {
            "statusCode": 200,
            "body": json.dumps(body)
        }

        return response

    except Exception as e:
        error_message = f"Error applying overlay: {str(e)}"
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": error_message})
        }
        return response
