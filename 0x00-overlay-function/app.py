from flask import Flask, request, jsonify
from flask_cors import CORS
from botocore.exceptions import ClientError
import boto3
import overlayModule
import json
import os
from urllib.parse import unquote

app = Flask(__name__)
CORS(app)


@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

def file_exists(image_name):
    try:
        # Create an S3 client
        s3 = boto3.client('s3')
        
        s3_bucket_name = 'adas-project-bucket'
        file_key = f'results/{image_name}.jpg'
        
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

@app.route('/apply_overlay', methods=['GET'])
def apply_overlay():
    try:
        # Extract parameters from query parameters
        image_name = request.args.get('image_name', '')
        image_type = request.args.get('image_type', '')
        res = file_exists(image_name)
        if (res==1):
            image_bucket_name = 'adas-project-bucket'
            mask_bucket_name = 'adas-project-bucket'
            img_file_path = f'./{unquote(image_name)}.jpg'
            mask_file_path = f'./{unquote(image_name)}.png'
            image_file_key = f'BDD-dataset/images/100k/{image_type}/{unquote(image_name)}.jpg'
            mask_file_key = f'BDD-dataset/labels/drivable/colormaps/{image_type}/{unquote(image_name)}.png'
            download_from_s3(image_bucket_name, image_file_key,img_file_path)
            download_from_s3(mask_bucket_name, mask_file_key,mask_file_path)
            response = overlayModule.overlay(image_name, img_file_path, mask_file_path)
            os.remove(img_file_path)
            os.remove(mask_file_path)

        else:
            s3_file_key = f'results/{image_name}.jpg'
            output_url = f'https://ddx0brhffx34i.cloudfront.net/{s3_file_key}'
            body = {
                "message": "Image already processed!",
                "data": {
                    "name": f'{image_name}.jpg',
                    "url": output_url
                },
            }

            response = {
                "statusCode": 200,
                "body": json.dumps(body)
            }
        return jsonify(response)

    except Exception as e:
        error_message = f"Error applying overlay: {str(e)}"
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": error_message})
        }
        return response

if __name__ == '__main__':
    app.run(debug=True)
