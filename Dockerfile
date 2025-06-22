# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# EXPOSE 8080 is good practice for documentation, but not strictly required by Cloud Run
EXPOSE 8080

# --- THE CRITICAL FIX IS HERE ---
# Use the PORT environment variable set by Cloud Run.
# The `sh -c` allows us to use the $PORT variable.
CMD sh -c "streamlit run app.py --server.port $PORT"