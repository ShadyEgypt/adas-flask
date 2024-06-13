# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY segmentModule.py /app/
COPY models/road-segmentation-adas-0001.xml /app/models/
COPY models/road-segmentation-adas-0001.bin /app/models/
COPY data /app/data/

# Make port 5008 available to the world outside this container
EXPOSE 5008

# Run app.py when the container launches
CMD ["python", "app.py"]
