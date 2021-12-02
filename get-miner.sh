#!/bin/bash

set -eu

if ! command -v nvidia-smi &> /dev/null
then 
    echo "Please install nvidia driver first"
    exit 1
fi

echo "Installing build-essential, python3-pip and nvidia-cuda-toolkit"
sudo apt install -y build-essential nvidia-cuda-toolkit cmake

echo "Installing conan"
temp_file=$(mktemp --suffix=.deb)
curl -L https://github.com/conan-io/conan/releases/latest/download/conan-ubuntu-64.deb -o $temp_file
sudo apt install $temp_file
rm -f $temp_file

echo "Git cloning gpu-miner"
git clone https://github.com/alephium/gpu-miner.git

echo "Building the gpu miner"
./gpu-miner/make.sh

echo "Your miner is built, you could run it with: gpu-miner/run-miner.sh"
