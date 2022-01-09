# gpu-miner

CUDA capable PoW miner for Alephium.

Please make sure that you have installed Nvidia driver for you GPU. You could verify that by running the `nvidia-smi` command.

### Ubuntu miner from source code

1. Build the miner by running

  ```shell
  curl -L https://github.com/alephium/gpu-miner/raw/master/get-miner.sh | bash
  ```

2. Start the miner

  ```shell
  gpu-miner/run-miner.sh
  ```

You could specify the miner api with `-a broker_ip` and `-p broker_port` parameters, GPU indexes with `-g 1 2`.

### Windows miner from source code

1. Install [Visual Studio Build Tools 2019](https://visualstudio.microsoft.com/vs/older-downloads/#visual-studio-2019-and-other-products), making sure to select [C++ CMake tools for Windows](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170#installation) during the installation.
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) (11.5 was tested and working)
3. Install [conan](https://docs.conan.io/en/latest/installation.html)
4. Build gpu-miner:
   1. Clone gpu-miner to local

   ``` shell
   git clone https://github.com/alephium/gpu-miner.git
   ```
   2. Open a powershell window, and launch the build script:

   ```shell
   cd your-gpu-miner-dir
   .\build.ps1
   ```
   Executable file will be generated in `your-gpu-miner-dir/bin/` directory.

5. Start the miner in a powershell window :
  ```shell
  .\run-miner.ps1
  ```

If you have any questions, please reach out to us on Discord.

### Pre-built miner

You could also download and run the pre-built miner from [Github release page](https://github.com/alephium/gpu-miner/releases). Note that your anti-virus might warn about the pre-built miner.
