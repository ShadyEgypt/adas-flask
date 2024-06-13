from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import segmentModule_onnx
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID_s3')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY_s3')
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

image_path = os.path.join('output_image.png')

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

def download_from_s3(bucket_name, file_key, local_file_path):
    # Download the file directly using the client
    s3.download_file(bucket_name, file_key, local_file_path)

def upload_to_s3(file_key, local_file_path):
    # Upload the resulting image to S3 with specified content type
    bucket_name = 'adas-project-bucket'
    s3.upload_file(
        Filename=local_file_path,
        Bucket=bucket_name,
        Key=file_key,
        ExtraArgs={
            'ContentType': 'image/png'
        }
    )

@app.route('/seg-yolov8-onnx-image', methods=['POST'])
def segment_image():
    try:
        data = request.json['body']
        # Extract parameters from the JSON object
        input_key = data.get('inputKey')
        name = data.get('name')   
        output_key = data.get('outputKey')
        image_bucket_name = 'adas-project-bucket'
        img_file_path = f'./{name}'
        print('content keys are extracted correctly')
        download_from_s3(image_bucket_name, input_key, img_file_path)
        print('file has been downloaded')
        response = segmentModule_onnx.segment(img_file_path)
        print(response)
        upload_to_s3(output_key, local_file_path=image_path)
        print('file has been uploaded')
        # os.remove(img_file_path)
        # os.remove('./output_image.png')
        output_url = f'https://ddx0brhffx34i.cloudfront.net/{output_key}'
        body = {
            "message": "Image uploaded successfully!",
            "data": {
                "name": f'segmented_{name}',
                "url": output_url
            },
        }

        response = {
            "statusCode": 200,
            "body": json.dumps(body)
        }
        return jsonify(response)

    except Exception as e:
        error_message = f"Error : {str(e)}"
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": error_message})
        }
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)

