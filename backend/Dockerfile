# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies, specifically FFmpeg for our audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend files (app.py, .pkl files, .pth file)
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]