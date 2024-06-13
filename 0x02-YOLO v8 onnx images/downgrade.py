import onnx
from onnx import version_converter, helper

# Load your original model
original_model = onnx.load("models/best.onnx")

# Convert model to a supported opset version
converted_model = version_converter.convert_version(original_model, 15)

# Save the converted model
onnx.save(converted_model, "models/converted.onnx")
