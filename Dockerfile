# Use official Python runtime as a parent image
FROM python:3.11-slim

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

# Command to run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}