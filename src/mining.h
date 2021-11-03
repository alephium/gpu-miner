#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include "fast.cu"

void worker_stream_callback(cudaStream_t stream, cudaError_t status, void *data);

void start_worker_mining(mining_worker_t *worker)
{
    cudaSetDevice(worker->device_id);

    reset_worker(worker);
    TRY( cudaMemcpyAsync(worker->device_hasher, worker->hasher, sizeof(blake3_hasher), cudaMemcpyHostToDevice, worker->stream) );

    cudaEvent_t startEvent, stopEvent;
    TRY( cudaEventCreate(&startEvent) );
    TRY( cudaEventCreate(&stopEvent) );

    TRY( cudaEventRecord(startEvent, worker->stream) );
    // blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    worker->grid_size = 9;
    worker->block_size = 384;
    blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    TRY( cudaEventRecord(stopEvent, worker->stream) );

    TRY( cudaMemcpyAsync(worker->hasher, worker->device_hasher, sizeof(blake3_hasher), cudaMemcpyDeviceToHost, worker->stream) );
    TRY( cudaStreamAddCallback(worker->stream, worker_stream_callback, worker, 0) );

    float time;
    TRY( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    // printf(" === mining time: %f\n", time);
}

#endif // ALEPHIUM_MINING_H
