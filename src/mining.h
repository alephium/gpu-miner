#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include "blake3.cu"
#include "time.h"

void start_worker_mining(mining_worker_t *worker)
{
    // clock_t start = clock();
    // printf("start mine: %d %d\n", work->job->from_group, work->job->to_group);
    reset_worker(worker);

    blake3_hasher_mine<<<32, 32, 32 * sizeof(blake3_hasher), worker->stream>>>(worker->hasher);
    TRY( cudaStreamSynchronize(worker->stream) );
    if (worker->hasher->found_good_hash) {
        store_worker_found_good_hash(worker, true);
    }
    // clock_t end = clock();
    // printf("=== mining time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
}

#endif // ALEPHIUM_MINING_H
