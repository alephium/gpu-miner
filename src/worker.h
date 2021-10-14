#ifndef ALEPHIUM_WORKER_H
#define ALEPHIUM_WORKER_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <atomic>

#include "messages.h"
#include "blake3.cu"
#include "uv.h"
#include "template.h"

typedef struct mining_worker_t {
    uint32_t id;

    cudaStream_t stream;
    blake3_hasher *hasher;
    blake3_hasher *device_hasher;

    std::atomic<bool> found_good_hash;
    std::atomic<mining_template_t *> template_ptr;
} mining_worker_t;

void mining_worker_init(mining_worker_t *self, uint32_t id)
{
    self->id = id;
    TRY( cudaStreamCreate(&(self->stream)) );
    TRY( cudaMallocHost(&(self->hasher), sizeof(blake3_hasher)) );
    TRY( cudaMalloc(&(self->device_hasher), sizeof(blake3_hasher)) );
    bzero(self->hasher->hash, 64);
}

bool load_worker__found_good_hash(mining_worker_t *worker)
{
    return atomic_load(&(worker->found_good_hash));
}

void store_worker_found_good_hash(mining_worker_t *worker, bool value)
{
    atomic_store(&(worker->found_good_hash), value);
}

mining_template_t *load_worker__template(mining_worker_t *worker)
{
    return atomic_load(&(worker->template_ptr));
}

void store_worker__template(mining_worker_t *worker, mining_template_t *template_ptr)
{
    atomic_store(&(worker->template_ptr), template_ptr);
}

void reset_worker(mining_worker_t *worker)
{
    blake3_hasher *hasher = worker->hasher;
    for (int i = 0; i < 24; i++) {
        hasher->buf[i] = rand();
    }
    mining_template_t *template_ptr = worker->template_ptr.load();
    job_t *job = template_ptr->job;
    memcpy(hasher->buf + 24, job->header_blob.blob, job->header_blob.len);
    hasher->buf_len = 24 + job->header_blob.len;
    memcpy(hasher->target, job->target.blob, job->target.len);
    hasher->target_len = job->target.len;
    hasher->from_group = job->from_group;
    hasher->to_group = job->to_group;
    hasher->hash_count = 0;
    hasher->found_good_hash = false;

    store_worker_found_good_hash(worker, false);
}

typedef struct mining_req {
    std::atomic<mining_worker_t *> worker;
} mining_req_t;

uv_work_t req[parallel_mining_works] = {};
mining_worker_t mining_workers[parallel_mining_works];
uint8_t write_buffers[parallel_mining_works][2048 * 1024];

void mining_workers_init()
{
    for (size_t i = 0; i < parallel_mining_works; i++) {
        mining_worker_t *worker = mining_workers + i;
        mining_worker_init(worker, (uint32_t)i);
    }
}

mining_worker_t *load_req_worker(uv_work_t *req)
{
    mining_req_t *mining_req = (mining_req_t *)req->data;
    return atomic_load(&(mining_req->worker));
}

void store_req_data(ssize_t worker_id, mining_worker_t *worker)
{
    if (!req[worker_id].data) {
        req[worker_id].data = malloc(sizeof(mining_req_t));
    }
    mining_req_t *mining_req = (mining_req_t *)(req[worker_id].data);
    atomic_store(&(mining_req->worker), worker);
}

ssize_t write_new_block(mining_worker_t *worker)
{
    uint32_t worker_id = worker->id;
    job_t *job = load_worker__template(worker)->job;
    uint8_t *nonce = worker->hasher->buf;
    uint8_t *write_pos = write_buffers[worker_id];

    ssize_t block_size = 24 + job->header_blob.len + job->txs_blob.len;
    ssize_t message_size = 1 + 4 + block_size;

    printf("message: %ld\n", message_size);
    write_size(&write_pos, message_size);
    write_byte(&write_pos, 0); // message type
    write_size(&write_pos, block_size);
    write_bytes(&write_pos, nonce, 24);
    write_blob(&write_pos, &job->header_blob);
    write_blob(&write_pos, &job->txs_blob);

    return message_size + 4;
}

void setup_template(mining_worker_t *worker, mining_template_t *template_ptr)
{
    add_template__ref_count(template_ptr, 1);
    store_worker__template(worker, template_ptr);
}

#endif // ALEPHIUM_WORKER_H
