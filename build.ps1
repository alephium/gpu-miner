$rootPath = Get-Location
$libuvPath = Join-Path $rootPath "libuv"
Set-Location $libuvPath
Remove-Item -Recurse -Force $libuvPath/build -ErrorAction Ignore
New-Item -Path $libuvPath -Name "build" -ItemType "directory" | Out-Null

# build libuv
$buildPath = Join-Path $libuvPath "build"
Set-Location $buildPath
cmd /c "cmake .."
Set-Location $libuvPath
cmd /c "cmake --build build --config Release"

# add include path and lib path
$includePath = Join-Path $libuvPath "include"
$env:Include += ";$includePath"
$libPath = Join-Path $buildPath "Release"
$env:Lib += ";$libPath"

# build gpu-miner
Set-Location $rootPath
cmd /c "nvcc --std c++11 -O3 --ptxas-options -v --x cu src/main.cu -lmsvcrt -luser32 -liphlpapi -luserenv -lws2_32 -luv_a -o bin/gpu-miner"
