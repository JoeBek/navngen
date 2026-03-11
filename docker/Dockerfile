# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /navngen

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* 

# Copy the requirements file and install dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/cvg/LightGlue.git /navngen/LightGlue

WORKDIR /navngen/LightGlue
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir .

WORKDIR /navngen

# Copy the rest of the application's source code
COPY . .

# Since the entrypoint is not defined, this container will start an interactive shell.
# You can run your scripts manually, e.g., python scripts/debug_traj.py
CMD [ "bash" ]
