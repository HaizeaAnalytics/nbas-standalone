# The final runtime image based on CUDA
FROM nvidia/cuda:11.0.3-base-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3-pip \
    gdal-bin \
    python3-gdal \
    libgdal-dev && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get install -y python3.11-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY /data data
COPY /connectivity connectivity
COPY /lookup_tables lookup_tables
COPY /tests tests

COPY main.py main.py

ENTRYPOINT ["/bin/sh", "-c"]    