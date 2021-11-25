# gpu-miner

CUDA capable PoW miner for Alephium.

Please make sure that you have installed Nvidia driver for you GPU. You could verify that by running the `nvidia-smi` command.

### Ubuntu miner from source code

1. Build the miner by running `curl -L https://github.com/alephium/gpu-miner/raw/master/get-miner.sh | bash`
2. Run `gpu-miner/run-miner.sh` to start the miner

You could specify the miner api with `-a broker` parameters, GPU indexes with `-g 1 2` and enable the alternate GPU init with '-o'. The alternate GPU init might improve performance for some cards.

### Windows miner from source code

1. Install Visual Studio Build Tools, Recommend [VS 2019](https://visualstudio.microsoft.com/vs/older-downloads/#visual-studio-2019-and-other-products)
2. Install [CUDA Toolkits](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
3. Build gpu-miner:
   1. Clone gpu-miner to local

   ``` sh
   git clone --recurse-submodules https://github.com/alephium/alephium.git
   ```
   2. Open **x64** Native Tools Command Prompt and execute:

   ```sh
   cd your-gpu-miner-dir
   make gpu
   ```

Executable file will be generated in `your-gpu-miner-dir/bin/` directory.

If you have any questions, please reach out to us on Discord.

### Pre-built miner

You could also download and run the pre-built miner from [Github release page](https://github.com/alephium/gpu-miner/releases/tag/0.4.0). Note that your anti-virus might warn about the pre-built miner.
