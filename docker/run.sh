#!/bin/bash

# first grab the root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "using root ${ROOT_DIR}"

# use the following command
CMD=$1
echo "executing: $CMD "

# run on the following GPU
GPU=${2:-0}
echo "using GPU: $GPU"

# check if we have a port arg
PORT=${3:-0}

# if [ -z ${PORT+x} ]; then
if [ $PORT == 0 ]; then
    # execute it in docker
    nvidia-docker run --ipc=host \
                  -v $HOME/datasets:/datasets \
                  -v /experiments/models:/models \
                  -v /experiments/logs:/logs \
                  -v ${ROOT_DIR}:/workspace \
                  -e NVIDIA_VISIBLE_DEVICES=$GPU \
                  -it jramapuram/pytorch:1.6.0-cuda10.1 $CMD ;
else
    # share the requested ports
    echo "exposing port: $PORT"

    # execute it in docker
    nvidia-docker run --ipc=host \
                  -v $HOME/datasets:/datasets \
                  -v /experiments/models:/models \
                  -v /experiments/logs:/logs \
                  -v ${ROOT_DIR}:/workspace \
                  -p $PORT:$PORT \
                  -e NVIDIA_VISIBLE_DEVICES=$GPU \
                  -it jramapuram/pytorch:1.6.0-cuda10.1 $CMD ;
fi
