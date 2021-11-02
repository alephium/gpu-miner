# gpu-miner

CUDA capable PoW miner for Alephium.

Please make sure that you have installed Nvidia driver for you GPU. You could verify that by running the `nvidia-smi` command.

## Usage

1. Install libuv and cuda toolkit. On Ubuntu, you could install them as follows:

    ``` sh
    sudo apt install libuv1-dev
    sudo apt install nvidia-cuda-toolkit
    ```

2. Run `make gpu` to build the executable miner. The output will be `bin/gpu-miner`
3. Run `bin/gpu-miner` and make sure that your full node is running

If you want to restart you miner automatically in case it crashes, please run the miner with the following command:

``` sh
until bin/gpu-miner; do
    echo "Miner crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
```

If you have any questions, please reach out to us on Discord.
