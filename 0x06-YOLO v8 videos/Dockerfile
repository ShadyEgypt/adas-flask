# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY segmentModule.py /app/
COPY models/best.pt /app/models/
COPY data /app/data/

# Make port 5006 available to the world outside this container
EXPOSE 5006

# Run app.py when the container launches
CMD ["python", "app.py"]
