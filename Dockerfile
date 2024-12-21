FROM python:3.10-slim
# Use an official Python runtime as a parent image
# FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Rust and Cargo
RUN apt-get update && apt-get install -y curl && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="/root/.cargo/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ML_Final_Project_Yashaswini:app", "--host", "0.0.0.0", "--port", "8000"]
