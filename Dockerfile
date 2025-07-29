# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files to the working directory
COPY . .

# Use PORT environment variable from Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}