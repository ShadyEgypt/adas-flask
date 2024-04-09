from flask import Flask, request, jsonify
from flask_cors import CORS
from botocore.exceptions import ClientError
import boto3
import segmentModule_onnx
import json
import os
from urllib.parse import unquote

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

def file_exists(name):
    try:
        # Create an S3 client
        s3 = boto3.client('s3')
        
        s3_bucket_name = 'adas-project-bucket'
        file_key = f'results/{name}'
        
        # Try to head the object to check if it exists
        response = s3.head_object(Bucket=s3_bucket_name, Key=file_key)
        print(f"File '{file_key}' exists in bucket '{s3_bucket_name}'.")
        return 0
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"File '{file_key}' does not exist in bucket '{s3_bucket_name}'.")
            return 1
        else:
            print(f"Error checking file existence: {e}")
            return 1


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

@app.route('/seg-img', methods=['POST'])
def segment_image():
    try:
        data = request.json
        # Extract parameters from the JSON object
        s3Key = data.get('s3Key')
        name = data.get('name')   
        outputS3Key = data.get('outputS3Key')
        image_bucket_name = 'adas-project-bucket'
        img_file_path = f'./{name}'
        image_file_key = f'{s3Key}'
        download_from_s3(image_bucket_name, image_file_key,img_file_path)
        response = segmentModule_onnx.segment(img_file_path)
        print(response)
        upload_to_s3(outputS3Key, local_file_path='./result.jpg');
        os.remove(img_file_path)
        os.remove('./result.jpg')
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
    app.run(debug=True, port=5002)
