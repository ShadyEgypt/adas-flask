# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY segmentModule_onnx.py /app/
COPY models/converted.onnx /app/models/

# Make port 5002 available to the world outside this container
EXPOSE 5002

# Run app.py when the container launches
CMD ["python", "app.py"]
