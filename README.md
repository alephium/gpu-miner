# gpu-miner

CUDA capable PoW miner for Alephium.

Please make sure that you have installed Nvidia driver for you GPU. You could verify that by running the `nvidia-smi` command.

## Usage

### Ubuntu

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

### Windows

1. Install Visual Studio Build Tools, Recommend [VS 2019](https://visualstudio.microsoft.com/vs/older-downloads/#visual-studio-2019-and-other-products)
2. Download [CUDA Toolkits](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
3. Build gpu-miner:
   1. Clone gpu-miner to local

   ``` sh
   git clone git@github.com:alephium/gpu-miner.git
   cd gpu-miner
   git checkout build-on-windows
   git submodule update --init --recursive
   ```
   2. Open **x64** Native Tools Command Prompt
   3. Execute:
   
   ```sh
   cd your-gpu-miner-dir
   powershell ./build.ps1
   ```

Executable file will be generated in `your-gpu-miner-dir/bin/` directory.

If you have any questions, please reach out to us on Discord.
