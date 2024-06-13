from flask import Flask, request, jsonify
from flask_cors import CORS
from botocore.exceptions import ClientError
import boto3
import segmentModule
import json
import os
from urllib.parse import unquote

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
image_path = './result.jpg'


@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

def download_from_s3(bucket_name, file_key, local_file_path): 
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)  
    # Download the file
    bucket.download_file(file_key, local_file_path)

def upload_to_s3(file_key, local_file_path): 
    # Upload the resulting image to S3
    bucket_name = 'adas-project-bucket'
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, file_key)
    # os.remove('result.jpg')

@app.route('/seg-adas2-image', methods=['POST'])
def segment():
    try:
        data = request.json
        # Extract parameters from the JSON object
        s3Key = data.get('s3Key')
        name = data.get('name')   
        outputS3Key = data.get('outputS3Key')
        model= data.get('model')
        image_bucket_name = 'adas-project-bucket'
        img_file_path = f'./{name}'
        image_file_key = f'{s3Key}'
        download_from_s3(image_bucket_name, image_file_key,img_file_path)
        response = segmentModule.segment(img_file_path, model)
        print(response)
        upload_to_s3(outputS3Key, local_file_path=image_path);
        os.remove(image_path)
        os.remove(img_file_path)
        output_url = f'https://ddx0brhffx34i.cloudfront.net/{outputS3Key}'
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
    app.run(debug=True, port=5004)
