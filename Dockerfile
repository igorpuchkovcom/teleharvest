# Use the official Python image from the Docker Hub
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the application
CMD ["python", "main.py"]
