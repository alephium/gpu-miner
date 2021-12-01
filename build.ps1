# requires powershell3 or later

cd $PSScriptRoot

mkdir -p build
cd build >$null 2>&1
# Specify the x64 architecture, otherwise can't find cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release --target gpu-miner
cd ..
