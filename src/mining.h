#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include "blake3.cu"
#include "time.h"

void start_worker_mining(mining_worker_t *worker)
{
    clock_t start = clock();
    reset_worker(worker);

    cudaEvent_t startEvent, stopEvent;
    TRY( cudaEventCreate(&startEvent) );
    TRY( cudaEventCreate(&stopEvent) );

    TRY( cudaEventRecord(startEvent, worker->stream) );
    blake3_hasher_mine<<<32, 32, 32 * sizeof(blake3_hasher), worker->stream>>>(worker->hasher);
    TRY( cudaEventRecord(stopEvent, worker->stream) );
    TRY( cudaStreamSynchronize(worker->stream) );

    float time;
    TRY( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf(" === mining time: %f\n", time);

    if (worker->hasher->found_good_hash) {
        store_worker_found_good_hash(worker, true);
    }
    clock_t end = clock();
    printf("=== mining time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
}

#endif // ALEPHIUM_MINING_H
