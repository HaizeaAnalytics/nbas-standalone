# Run standalone NBAS in docker container (linux)

## Requirements
- [CUDA enabled GPU with compute capability > 8](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/)


### 1. Download docker image
`sudo docker image pull haizeaanalytics/nbas:latest`

### 2. Run docker image and mount the home directory
`sudo docker run -ti --gpus all -v $HOME:/data haizeaanalytics/nbas:latest /bin/bash`

*Note: This opens an interactive session inside the container with your home directory mounted in /data. Please ensure your payload is available in this directory. An example payload can be found [here](https://raw.githubusercontent.com/chamith-ed/nbas-standalone/refs/heads/main/tests/test_payload.json).*

### 3. Run NBAS in the docker container
`python3.11 main.py /data/path/to/payload.json`

## Disclaimer
Results from this standalone version of NBAS differ slightly from the original NBAS service due to slight differences in how areas are clipped from the full datasets.
