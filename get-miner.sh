#!/bin/bash

set -eu

if ! command -v nvidia-smi &> /dev/null
then 
    echo "Please install nvidia driver first"
    exit 1
fi

echo "Installing build-essential libuv nvidia-cuda-toolkit"
sudo apt install -y build-essential libuv1-dev nvidia-cuda-toolkit

echo "Git cloning gpu-miner"
git clone https://github.com/alephium/gpu-miner.git

echo "Building the gpu miner"
cd gpu-miner && make linux-gpu

echo "Your miner is built, you could find it at gpu-miner/bin/gpu-miner"