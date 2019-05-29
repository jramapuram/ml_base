#!/bin/bash

# first grab the root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "using root ${ROOT_DIR}"

# use the following command
CMD=$1
echo "executing $CMD "

# execute it in docker
nvidia-docker run --ipc=host -v $HOME/datasets:/datasets -v ${ROOT_DIR}:/workspace -it jramapuram/pytorch:1.1.0-cuda10.0 $CMD
