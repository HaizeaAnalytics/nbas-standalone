# The final runtime image based on CUDA
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    gdal-bin \
    libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

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