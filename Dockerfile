# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
RUN apt update && apt upgrade -y && \
    apt install -y wget && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    google-chrome --version

WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Run uvicorn server
# --host 0.0.0.0 makes the server accessible from outside the container
# --reload enables auto-reloading for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]