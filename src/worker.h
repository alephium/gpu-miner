#ifndef ALEPHIUM_WORKER_H
#define ALEPHIUM_WORKER_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <atomic>
#include <random>

#include "messages.h"
#include "uv.h"
#include "template.h"
#include "opencl_util.h"

typedef struct
{
    uint8_t buf[BLAKE3_BUF_CAP];

    uint32_t cv[8];

    uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

    uint8_t target[32];
    uint32_t from_group;
    uint32_t to_group;

    uint32_t hash_count;
    int found_good_hash;
} blake3_hasher;

typedef struct mining_worker_t {
    bool on_service = false;
    cl_uint platform_index;
    cl_platform_id platform_id;
    cl_uint device_index;
    cl_device_id device_id;
    int grid_size;
    int block_size;

    blake3_hasher *hasher;
    blake3_hasher *device_hasher;

    std::atomic<bool> found_good_hash;
    std::atomic<mining_template_t *> template_ptr;

    std::mt19937 random_gen;

    uv_async_t async;
    uv_timer_t timer;
} mining_worker_t;

void mining_worker_init(mining_worker_t *self, cl_uint platform_index, cl_platform_id platform_id, cl_uint device_index, cl_device_id device_id)
{
    self->on_service = true;
    self->platform_index = platform_index;
    self->platform_id = platform_id;
    self->device_index = device_index;
    self->device_id = device_id;

    // cudaSetDevice(device_id);
    // TRY( cudaStreamCreate(&(self->stream)) );
    // config_cuda(device_id, &self->grid_size, &self->block_size);
    // printf("Worker %d: device id %d, grid size %d, block size %d\n", self->id, self->device_id, self->grid_size, self->block_size);

    // TRY( cudaMallocHost(&(self->hasher), sizeof(blake3_hasher)) );
    // TRY( cudaMalloc(&(self->device_hasher), sizeof(blake3_hasher)) );
    self->hasher = (blake3_hasher *)malloc(sizeof(blake3_hasher));
    memset(self->hasher->buf, 0, BLAKE3_BUF_CAP);
    memset(self->hasher->hash, 0, 64);

    self->random_gen.seed((uint64_t)self + (uint64_t)self->hasher + rand());
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
    std::uniform_int_distribution<> distrib(0, UINT8_MAX);
    blake3_hasher *hasher = worker->hasher;
    for (int i = 0; i < 24; i++) {
        hasher->buf[i] = distrib(worker->random_gen);
    }

    mining_template_t *template_ptr = worker->template_ptr.load();
    job_t *job = template_ptr->job;
    memcpy(hasher->buf + 24, job->header_blob.blob, job->header_blob.len);
    assert((24 + job->header_blob.len) == BLAKE3_BUF_LEN);
    assert((24 + job->header_blob.len + 63) / 64 * 64 == BLAKE3_BUF_CAP);

    size_t target_zero_len = 32 - job->target.len;
    memset(hasher->target, 0, target_zero_len);
    memcpy(hasher->target + target_zero_len, job->target.blob, job->target.len);

    hasher->from_group = job->from_group;
    hasher->to_group = job->to_group;

    hasher->hash_count = 0;
    hasher->found_good_hash = false;

    store_worker_found_good_hash(worker, false);
}

typedef struct mining_req {
    std::atomic<mining_worker_t *> worker;
} mining_req_t;

uv_work_t req[max_platform_num][max_gpu_num][parallel_mining_works_per_gpu] = { NULL };
mining_worker_t mining_workers[max_platform_num][max_gpu_num][parallel_mining_works_per_gpu];

mining_worker_t *load_req_worker(uv_work_t *req)
{
    mining_req_t *mining_req = (mining_req_t *)req->data;
    return atomic_load(&(mining_req->worker));
}

void store_req_data(cl_uint platform_index, cl_uint device_index, size_t worker_id, mining_worker_t *worker)
{
    uv_work_t _req = req[platform_index][device_index][worker_id];
    if (!_req.data) {
        _req.data = malloc(sizeof(mining_req_t));
    }
    mining_req_t *mining_req = (mining_req_t *)(_req.data);
    atomic_store(&(mining_req->worker), worker);
}

void mining_workers_init(cl_uint platform_index, cl_platform_id platform_id, cl_uint device_index, cl_device_id device_id)
{
    for (size_t i = 0; i < parallel_mining_works_per_gpu; i++) {
        mining_worker_t *worker = &(mining_workers[platform_index][device_index][i]);
        mining_worker_init(worker, platform_index, platform_id, device_index, device_id);
        store_req_data(platform_index, device_index, i, worker);
    }
}

ssize_t write_new_block(mining_worker_t *worker, uint8_t *write_buf)
{
    job_t *job = load_worker__template(worker)->job;
    uint8_t *nonce = worker->hasher->buf;
    uint8_t *write_pos = write_buf;

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
