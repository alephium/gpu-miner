#ifndef ALEPHIUM_BLAKE3_CU
#define ALEPHIUM_BLAKE3_CU

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "constants.h"
#include "messages.h"


// Include both blake implementations
#include "blake3/inlined-blake.hpp"
#include "blake3/original-blake.hpp"

#ifdef BLAKE3_TEST
#include <cuda_profiler_api.h>
int main()
{
    cudaProfilerStart();
    blob_t target;
    hex_to_bytes("00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff", &target);

    inline_blake::blake3_hasher *hasher;
    inline_blake::blake3_hasher *device_hasher;
    TRY(cudaMallocHost(&hasher, sizeof(inline_blake::blake3_hasher)));
    TRY(cudaMalloc(&device_hasher, sizeof(inline_blake::blake3_hasher)));

    bzero(hasher->buf, BLAKE3_BUF_CAP);
    memcpy(hasher->target, target.blob, target.len);
    hasher->from_group = 0;
    hasher->to_group = 3;

    cudaStream_t stream;
    TRY(cudaStreamCreate(&stream));
    TRY(cudaMemcpyAsync(device_hasher, hasher, sizeof(blake3_hasher), cudaMemcpyHostToDevice, stream));
    inline_blake::blake3_hasher_mine<<<10, 1024, 0, stream>>>(device_hasher);
    TRY(cudaStreamSynchronize(stream));

    TRY(cudaMemcpy(hasher, device_hasher, sizeof(blake3_hasher), cudaMemcpyDeviceToHost));
    char *hash_string1 = bytes_to_hex(hasher->hash, 32);
    printf("good: %d\n", hasher->found_good_hash);
    printf("nonce: %d\n", hasher->buf[0]);
    printf("count: %d\n", hasher->hash_count);
    printf("%s\n", hash_string1); // 0003119e5bf02115e1c8496008fbbcec4884e0be7f9dc372cd4316a51d065283
    cudaProfilerStop();
}
#endif // BLAKE3_TEST

// Beginning of GPU Architecture definitions
inline int get_sm_cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int get_device_cores(int device_id)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    int cores_size = get_sm_cores(props.major, props.minor) * props.multiProcessorCount;
    return cores_size;
}

void config_cuda(int device_id, int *grid_size, int *block_size, bool* is_inline_miner)
{
    cudaSetDevice(device_id);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    // If using a 2xxx or 3xxx card, use the new grid calc
    bool use_rtx_grid_bloc = ((props.major << 4) + props.minor) >= 0x75;
    
    // If compiling for windows, override the test and force the new calc
#ifdef _WIN32
    use_rtx_grid_bloc = true;
#endif

    // If compiling for linux, and we're not using the RTX grid block, force the original miner, otherwise use the inlined one
#ifdef __linux__
    *is_inline_miner = use_rtx_grid_bloc;
#else
    *is_inline_miner = true;
#endif
    if(*is_inline_miner) {
        cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, inline_blake::blake3_hasher_mine);
    } else {
        cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, ref_blake::blake3_hasher_mine);
    }
    
    int cores_size = get_device_cores(device_id);
    if (use_rtx_grid_bloc) {
        *grid_size = props.multiProcessorCount * 2;
        *block_size = cores_size / *grid_size * 4;
    } else {
        *grid_size = cores_size / *block_size * 3 / 2;
    }
}

#endif // ALEPHIUM_BLAKE3_CU
