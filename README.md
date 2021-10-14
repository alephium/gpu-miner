# gpu-miner

CUDA capable PoW miner for Alephium.

## Usage

1. Install libuv and cuda toolkit. On Ubuntu, you could install them as follows:

    ``` sh
    sudo apt install libuv1-dev
    sudo apt install nvidia-cuda-toolkit
    ```

2. Run `make gpu` to build the executable miner. The output will be `bin/gpu-miner`
3. Run `bin/gpu-miner` and make sure that your full node is running

If you have any questions, please reach out to us on Discord.