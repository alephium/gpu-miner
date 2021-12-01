#!/bin/bash

set -eu

if ! command -v nvidia-smi &> /dev/null
then 
    echo "Please install nvidia driver first"
    exit 1
fi

echo "Installing build-essential, python3-pip and nvidia-cuda-toolkit"
sudo apt install -y build-essential python3-pip nvidia-cuda-toolkit

echo "Installing conan"
python3 -m pip install conan

echo "Git cloning gpu-miner"
git clone https://github.com/alephium/gpu-miner.git

echo "Building the gpu miner"
chmod +x ./gpu-miner/make.sh && ./gpu-miner/make.sh

echo "Your miner is built, you could run it with: gpu-miner/run-miner.sh"