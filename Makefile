gpu:
	@nvcc --std c++11 -O3 --ptxas-options -v --x cu src/main.cu -luv -o bin/gpu-miner

blake3-test:
	@nvcc -DBLAKE3_TEST --std c++11 --x cu src/fast.cu -luv -o bin/blake3-test && bin/blake3-test
