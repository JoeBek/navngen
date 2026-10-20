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

# Pre-download the SuperPoint + LightGlue weights into the image so runs need
# no network access (torch.hub cache lives at /root/.cache/torch).
RUN python -c "from lightglue import SuperPoint, LightGlue; \
    SuperPoint(max_num_keypoints=256).eval(); \
    LightGlue(features='superpoint').eval()"

# Copy the rest of the application's source code
COPY . .

# Run the trajectory pipeline by default. Arguments passed to `docker run` are
# forwarded to the script, e.g.:
#   docker run --rm -v /data:/data navngen \
#     --input_path /data/airport --config_path scripts/configs/camera_forward.yaml \
#     --config_type surfnav --image_dirname images --output_path /data/out.tum
ENTRYPOINT [ "python", "scripts/get_trajectory.py" ]
