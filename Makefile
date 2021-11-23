ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    target := windows-gpu
else
    target := linux-gpu  # same as "uname -s"
endif

gpu: $(target)

windows-gpu:
	@powershell ./build.ps1

linux-gpu:
	@g++ --std c++11 src/main.cc -luv -lOpenCL -o bin/amd-miner && bin/amd-miner

blake3-test:
	@nvcc -DBLAKE3_TEST --std c++11 --x cu src/blake3.cu -luv -o bin/blake3-test && bin/blake3-test
