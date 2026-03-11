#!/bin/bash

# Build the Docker image
#docker build -t navngen -f docker/Dockerfile .

DOCKER_ARGS=""

#DOCKER_ARGS+="-v /home/joe/vt/LightGlue/"
DOCKER_ARGS+="-v /home/joe/vt/research/glue/data/airport/cropped_images:/data/airport/"
DOCKER_ARGS+=" -v /home/joe/vt/research/glue/SurfNav4UAS_Data/config:/config/surfnav/"
# Run the Docker container
# This will start an interactive session inside the container.
# You can then run your python scripts.
docker run -it --rm \
    -v "$(pwd)/src:/app/src" \
    $DOCKER_ARGS \
    navngen
