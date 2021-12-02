#!/bin/bash

# Bash script to build miner

cd $(dirname "$0")

echoerr() { echo "$@" 1>&2; }

# cmake dep
if ! command -v cmake &> /dev/null
then
    echo -n "CMake could not be found, trying to install automatically..."
    # Try installing
    (sudo apt-get update >/dev/null && sudo apt-get -y install cmake >/dev/null) || ((echoerr "Could not install cmake, exiting") && exit)
    # Die if not found
    (command -v cmake &>/dev/null) || ((echoerr "Installed CMake not found, exiting") && exit)
    echo "Success !"
fi

# conan dep
if ! command -v conan &> /dev/null
then
    echo -n "conan could not be found, trying to install automatically..."
    # Try installing
    (temp_file=$(mktemp --suffix=.deb) && \
     curl -s -L https://github.com/conan-io/conan/releases/latest/download/conan-ubuntu-64.deb -o $temp_file && \
     sudo apt-get -y  install $temp_file >/dev/null) || ((echoerr "Could not install conan, exiting") && exit)
    # Die if not found
    (command -v cmake &>/dev/null) || ((echoerr "Installed conan not found, exiting") && exit)
    echo "Success !"
fi

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
cmake --build . --config Release --target gpu-miner
