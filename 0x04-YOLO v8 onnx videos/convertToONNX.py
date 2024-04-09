from ultralytics import YOLO

model = YOLO('./best.pt')
output_model_path = 'best.xml'
model.export(format='openvino', dynamic=True, half=False)