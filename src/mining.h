#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include "blake3.cu"

void worker_stream_callback(cudaStream_t stream, cudaError_t status, void *data);

void start_worker_mining(mining_worker_t *worker)
{
    cudaSetDevice(worker->device_id);

    reset_worker(worker);
    if (worker->is_inline_miner){
        TRY( cudaMemcpyAsync(worker->device_hasher.inline_hasher, worker->host_hasher.inline_hasher, sizeof(inline_blake::blake3_hasher), cudaMemcpyHostToDevice, worker->stream) );
    } else{
        TRY( cudaMemcpyAsync(worker->device_hasher.ref_hasher, worker->host_hasher.ref_hasher, sizeof(ref_blake::blake3_hasher), cudaMemcpyHostToDevice, worker->stream) );
    }

    cudaEvent_t startEvent, stopEvent;
    TRY( cudaEventCreate(&startEvent) );
    TRY( cudaEventCreate(&stopEvent) );

    TRY( cudaEventRecord(startEvent, worker->stream) );
    // blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    if (worker->is_inline_miner) {
        inline_blake::blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher.inline_hasher);
    } else {
        ref_blake::blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher.inline_hasher);
    }
    TRY( cudaEventRecord(stopEvent, worker->stream) );

    if (worker->is_inline_miner) {
        TRY(cudaMemcpyAsync(worker->host_hasher.inline_hasher, worker->device_hasher.inline_hasher, sizeof(inline_blake::blake3_hasher), cudaMemcpyDeviceToHost, worker->stream));
    } else{
        TRY(cudaMemcpyAsync(worker->host_hasher.ref_hasher, worker->device_hasher.ref_hasher, sizeof(ref_blake::blake3_hasher), cudaMemcpyDeviceToHost, worker->stream));
    }

    TRY( cudaStreamAddCallback(worker->stream, worker_stream_callback, worker, 0) );

    float time;
    TRY( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    // printf(" === mining time: %f\n", time);
}

#endif // ALEPHIUM_MINING_H
