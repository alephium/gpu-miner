#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include <chrono>
#include "blake3.cu"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point_t;

void start_worker_mining(mining_worker_t *worker)
{
    cudaSetDevice(worker->device_id);

    time_point_t start = Time::now();
    reset_worker(worker);
    TRY( cudaMemcpyAsync(worker->device_hasher, worker->hasher, sizeof(blake3_hasher), cudaMemcpyHostToDevice, worker->stream) );

    cudaEvent_t startEvent, stopEvent;
    TRY( cudaEventCreate(&startEvent) );
    TRY( cudaEventCreate(&stopEvent) );

    TRY( cudaEventRecord(startEvent, worker->stream) );
    // blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    TRY( cudaEventRecord(stopEvent, worker->stream) );

    TRY( cudaMemcpyAsync(worker->hasher, worker->device_hasher, sizeof(blake3_hasher), cudaMemcpyDeviceToHost, worker->stream) );
    TRY( cudaStreamSynchronize(worker->stream) );

    float time;
    TRY( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    // printf(" === mining time: %f\n", time);

    if (worker->hasher->found_good_hash) {
        store_worker_found_good_hash(worker, true);
    }
    duration_t elapsed = Time::now() - start;
    // printf("=== mining time: %fs\n", elapsed.count());
}

#endif // ALEPHIUM_MINING_H
