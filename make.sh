#!/bin/bash

# Bash script to build miner

cd $(dirname "$0")

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
cmake --build . --config Release --target gpu-miner
