#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

#include "blake3.cu"

void start_worker_mining(mining_worker_t *worker)
{
    // printf("start mine: %d %d\n", work->job->from_group, work->job->to_group);
    reset_worker(worker);

    blake3_hasher_mine<<<32, 32, 32 * sizeof(blake3_hasher), worker->stream>>>(worker->hasher);
    TRY( cudaStreamSynchronize(worker->stream) );
    if (worker->hasher->found_good_hash) {
        store_worker_found_good_hash(worker, true);
    }
}

#endif // ALEPHIUM_MINING_H
