# ADAS project

This projects consists of two or three different flask servers, each of which will be containerized and pushed to AWS Fargate

## flask app 1
This app takes a url of an image on s3 and returns a url of the annotated image

- Original Image

<img alt="Input Image." src="assets/original.jpg">


- Annotated Image

<img alt="Output Image." src="assets/annotated.jpg">


## flask app 2
This app takes a url of an image on s3 and returns a url to the segmented image using YOLO v2


## flask app 3
This app takes a url of an image on s3 and returns a url to the segmented image using YOLO v8

- Original Image

<img alt="Input Image." src="assets/d3f34243-a7166713.jpg">


- Segmented Image

<img alt="Output Image." src="assets/segmented_d3f34243-a7166713.jpg">