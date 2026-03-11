# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake

# Copy the requirements file and install dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/cvg/LightGlue.git /app/LightGlue

WORKDIR /app/LightGlue
RUN pip install --no-cache-dir -r requirements.txt || true

WORKDIR /app

# Copy the rest of the application's source code
COPY . .

# Since the entrypoint is not defined, this container will start an interactive shell.
# You can run your scripts manually, e.g., python scripts/debug_traj.py
CMD [ "bash" ]
