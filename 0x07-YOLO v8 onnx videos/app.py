from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import segmentModule
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID_s3')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY_s3')
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

video_path = './output_video.mp4'
csv_path = os.path.join('output_csv.csv')

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

def download_from_s3(bucket_name, file_key, local_file_path):
    # Download the file directly using the client
    s3.download_file(bucket_name, file_key, local_file_path)

def upload_video(file_key, local_file_path): 
    # Upload the resulting video to S3
    bucket_name = 'adas-project-bucket'
    s3.upload_file(
        Filename=local_file_path,
        Bucket=bucket_name,
        Key=file_key,
        ExtraArgs={
            'ContentType': 'video/mp4'
        }
    )

def upload_csv(file_key, local_file_path):
    # Upload the resulting CSV file to S3
    bucket_name = 'adas-project-bucket'
    s3.upload_file(
        Filename=local_file_path,
        Bucket=bucket_name,
        Key=file_key,
        ExtraArgs={
            'ContentType': 'text/csv'
        }
    )

@app.route('/seg-yolov8-opt-video', methods=['POST'])
def segment():
    try:
        data = request.json['body']
        # Extract parameters from the JSON object
        input_key = data.get('inputKey')
        name = data.get('name')   
        output_key = data.get('outputKey')
        output_csv = data.get('outputCsv')
        video_bucket_name = 'adas-project-bucket'
        vid_file_path = f'./data/{name}'
        print('content keys are extracted correctly')
        download_from_s3(video_bucket_name, input_key, vid_file_path)
        print('file has been downloaded')
        response = segmentModule.segment(vid_file_path)        
        print(response)
        upload_csv(output_csv, local_file_path=csv_path)
        upload_video(output_key, local_file_path=video_path)
        os.remove(vid_file_path)
        os.remove(video_path)
        output_url = f'https://d33csf9naiv7sh.cloudfront.net/{output_key}'
        body = {
            "message": "Video uploaded successfully!",
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
    app.run(host='0.0.0.0', debug=True, port=5007)

