FROM nvidia/cuda:13.0.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
	build-essential \
	cmake \
	git \
	python3-dev \
	python3-numpy \
	libgtk2.0-dev \
	pkg-config \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	pip \
	python3.12-venv


RUN cd ~ && git clone https://github.com/opencv/opencv.git && \
	cd opencv && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D \
	CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j$(nproc) && make install && ldconfig && cd ~

# create python virtualenv
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"


RUN git clone https://github.com/cvg/LightGlue.git && cd LightGlue && \
	python3 -m pip install -e .


