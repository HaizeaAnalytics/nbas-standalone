# Run standalone NBAS in docker container (linux)

## Requirements
- [CUDA enabled GPU with compute capability > 8](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/)


### 1. Download docker image
`sudo docker image pull haizeaanalytics/nbas:latest`

### 2. Run docker image and mount the current directory
`sudo docker run -ti --gpus all -v ./:/data haizeaanalytics/nbas:latest /bin/bash`

*Note: This opens an interactive session inside the container with your current directory mounted. Please ensure your payload is available in this directory*

### 3. Run NBAS in the docker container
`python3.11 main.py /data/payload.json`

## Disclaimer
Results from this standalone version of NBAS differ slightly from the original NBAS service due to slight differences in how areas are clipped from the full datasets.
