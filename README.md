# Run standalone NBAS in docker container (linux)

## Requirements
- [CUDA enabled GPU with compute capability > 8](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/)


### 1. Download docker image
`sudo docker image pull chamithed/nbas:latest`

### 2. Run docker image. Keep this terminal window open - you'll need it later:
`sudo docker run -ti --gpus all chamithed/nbas:latest /bin/bash`

*Note: This opens an interactive session inside the container*

### 3. Open new terminal and find container name
`sudo docker container ls`

### 4. Copy payload into container using the container_name from the previous step
`sudo docker cp ./payload.json {container_name}:/app/payload.json`

*Note: An example payload can be found in tests/test_payload.json in this repo*

### 5. Switch back to your first terminal and run NBAS in the docker container
`python3.11 main.py payload.json`

### 6. Go back to the second terminal and copy result to host machine
`sudo docker cp {container_name}:/app/results/xxxx-results.json ./result.json`

## Disclaimer
Results from this standalone version of NBAS differ slightly from the original NBAS service due to slight differences in how areas are clipped from the full datasets.
