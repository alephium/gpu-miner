#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <chrono>
#include <mutex>

#include "constants.h"
#include "uv.h"
#include "messages.h"
#include "blake3.cu"
#include "pow.h"
#include "worker.h"
#include "template.h"
#include "mining.h"
#include "getopt.h"


std::atomic<uint32_t> found_solutions{0};

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point_t;

uv_loop_t *loop;
uv_stream_t *tcp;

time_point_t start_time = Time::now();

std::atomic<int> gpu_count;
std::atomic<int> worker_count;
std::atomic<uint64_t> total_mining_count;
std::atomic<uint64_t> device_mining_count[max_gpu_num];
bool use_device[max_gpu_num];

void setup_gpu_worker_count(int _gpu_count, int _worker_count)
{
    gpu_count.store(_gpu_count);
    worker_count.store(_worker_count);
}

void on_write_end(uv_write_t *req, int status)
{
    if (status < 0)
    {
        fprintf(stderr, "error on_write_end %d", status);
        exit(1);
    }
    free(req);
}

std::mutex write_mutex;
uint8_t write_buffer[4096 * 1024];
void submit_new_block(mining_worker_t *worker)
{
    expire_template_for_new_block(load_worker__template(worker));

    const std::lock_guard<std::mutex> lock(write_mutex);

    ssize_t buf_size = write_new_block(worker, write_buffer);
    uv_buf_t buf = uv_buf_init((char *)write_buffer, buf_size);
    print_hex("new solution", (uint8_t *) hasher_buf(worker, true), 32);

    uv_write_t *write_req = (uv_write_t *)malloc(sizeof(uv_write_t));
    uint32_t buf_count = 1;

    uv_write(write_req, tcp, &buf, buf_count, on_write_end);
    found_solutions.fetch_add(1, std::memory_order_relaxed);
}

void mine_with_timer(uv_timer_t *timer);

void mine(mining_worker_t *worker)
{
    time_point_t start = Time::now();

    int32_t to_mine_index = next_chain_to_mine();
    if (to_mine_index == -1)
    {
        printf("waiting for new tasks\n");
        worker->timer.data = worker;
        uv_timer_start(&worker->timer, mine_with_timer, 500, 0);
    } else {
        mining_counts[to_mine_index].fetch_add(mining_steps);
        setup_template(worker, load_template(to_mine_index));

        start_worker_mining(worker);

        duration_t elapsed = Time::now() - start;
        // printf("=== mining time: %fs\n", elapsed.count());
    }
}

void mine_with_req(uv_work_t *req)
{
    mining_worker_t *worker = load_req_worker(req);
    mine(worker);
}

void mine_with_async(uv_async_t *handle)
{
    mining_worker_t *worker = (mining_worker_t *)handle->data;
    mine(worker);
}

void mine_with_timer(uv_timer_t *timer)
{
    mining_worker_t *worker = (mining_worker_t *)timer->data;
    mine(worker);
}

void after_mine(uv_work_t *req, int status)
{
    return;
}

void worker_stream_callback(cudaStream_t stream, cudaError_t status, void *data)
{
    mining_worker_t *worker = (mining_worker_t *)data;
    if (hasher_found_good_hash(worker, true))
    {
        store_worker_found_good_hash(worker, true);
        submit_new_block(worker);
    }

    mining_template_t *template_ptr = load_worker__template(worker);
    job_t *job = template_ptr->job;
    uint32_t chain_index = job->from_group * group_nums + job->to_group;
    mining_counts[chain_index].fetch_sub(mining_steps);
    mining_counts[chain_index].fetch_add(hasher_hash_count(worker, true));
    total_mining_count.fetch_add(hasher_hash_count(worker, true));
    device_mining_count[worker->device_id].fetch_add(hasher_hash_count(worker, true));
    free_template(template_ptr);
    worker->async.data = worker;
    uv_async_send(&(worker->async));
}

void start_mining()
{
    assert(mining_templates_initialized == true);

    start_time = Time::now();

    for (uint32_t i = 0; i < worker_count.load(); i++)
    {
        if (use_device[mining_workers[i].device_id])
        {
            uv_queue_work(loop, &req[i], mine_with_req, after_mine);
        }
    }
}

void start_mining_if_needed()
{
    if (!mining_templates_initialized)
    {
        bool all_initialized = true;
        for (int i = 0; i < chain_nums; i++)
        {
            if (load_template(i) == NULL)
            {
                all_initialized = false;
                break;
            }
        }
        if (all_initialized)
        {
            mining_templates_initialized = true;
            start_mining();
        }
    }
}

void alloc_buffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf)
{
    buf->base = (char *)malloc(suggested_size);
    buf->len = suggested_size;
}

void log_hashrate(uv_timer_t *timer)
{
    time_point_t current_time = Time::now();
    if (current_time > start_time)
    {
        duration_t eplased = current_time - start_time;
        printf("hashrate: %.0f MH/s ", total_mining_count.load() / eplased.count() / 1000000);
        for (int i = 0; i < gpu_count; i++)
        {
            printf("gpu%d: %.0f MH/s ", i, device_mining_count[i].load() / eplased.count() / 1000000);
        }
        printf("solutions: %u\n", found_solutions.load(std::memory_order_relaxed));
    }
}

uint8_t read_buf[2048 * 1024 * chain_nums];
blob_t read_blob = {read_buf, 0};
server_message_t *decode_buf(const uv_buf_t *buf, ssize_t nread)
{
    if (read_blob.len == 0)
    {
        read_blob.blob = (uint8_t *)buf->base;
        read_blob.len = nread;
        server_message_t *message = decode_server_message(&read_blob);
        if (message)
        {
            // some bytes left
            if (read_blob.len > 0)
            {
                memcpy(read_buf, read_blob.blob, read_blob.len);
                read_blob.blob = read_buf;
            }
            return message;
        }
        else
        { // no bytes consumed
            memcpy(read_buf, buf->base, nread);
            read_blob.blob = read_buf;
            read_blob.len = nread;
            return NULL;
        }
    }
    else
    {
        assert(read_blob.blob == read_buf);
        memcpy(read_buf + read_blob.len, buf->base, nread);
        read_blob.len += nread;
        return decode_server_message(&read_blob);
    }
}

void on_read(uv_stream_t *server, ssize_t nread, const uv_buf_t *buf)
{
    if (nread < 0)
    {
        fprintf(stderr, "error on_read %ld: might be that the full node is not synced, or miner wallets are not setup\n", nread);
        exit(1);
    }

    if (nread == 0)
    {
        return;
    }

    server_message_t *message = decode_buf(buf, nread);
    if (message)
    {
        switch (message->kind)
        {
        case JOBS:
            for (int i = 0; i < message->jobs->len; i++)
            {
                update_templates(message->jobs->jobs[i]);
            }
            start_mining_if_needed();
            break;

        case SUBMIT_RESULT:
            printf("submitted: %d -> %d: %d \n", message->submit_result->from_group, message->submit_result->to_group, message->submit_result->status);
            break;
        }
        free_server_message_except_jobs(message);
    }

    free(buf->base);
    // uv_close((uv_handle_t *) server, free_close_cb);
}

void on_connect(uv_connect_t *req, int status)
{
    if (status < 0)
    {
        fprintf(stderr, "connection error %d: might be that the full node is not reachable\n", status);
        exit(1);
    }
    printf("the server is connected %d %p\n", status, req);

    tcp = req->handle;
    uv_read_start(req->handle, alloc_buffer, on_read);
}

bool is_valid_ip_address(char *ip_address)
{
    struct sockaddr_in sa;
    int result = inet_pton(AF_INET, ip_address, &(sa.sin_addr));
    return result != 0;
}

int hostname_to_ip(char *ip_address, char *hostname)
{
    struct addrinfo hints, *servinfo;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int res = getaddrinfo(hostname, NULL, &hints, &servinfo);
    if (res != 0)
    {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
        return 1;
    }

    struct sockaddr_in *h = (struct sockaddr_in *)servinfo->ai_addr;
    strcpy(ip_address, inet_ntoa(h->sin_addr));

    freeaddrinfo(servinfo);
    return 0;
}
#ifndef MINER_VERSION
#define MINER_VERSION "unknown"
#endif

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

    #ifdef _WIN32
    WSADATA wsa;
    // current winsocket version is 2.2
    int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (rc != 0)
    {
        printf("Initialize winsock failed: %d", rc);
        exit(1);
    }
    #endif

    printf("Running gpu-miner version : %s\n", MINER_VERSION);

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    printf("GPU count: %d\n", gpu_count);
    for (int i = 0; i < gpu_count; i++)
    {
        printf("GPU #%d has #%d cores\n", i, get_device_cores(i));
        use_device[i] = true;
    }

    int port = 10973;
    char broker_ip[16];
    strcpy(broker_ip, "127.0.0.1");

    int command;
    while ((command = getopt(argc, argv, "p:g:a:")) != -1)
    {
        switch (command)
        {
        case 'p':
            port = atoi(optarg);
            break;
        case 'a':
            if (is_valid_ip_address(optarg))
            {
                strcpy(broker_ip, optarg);
            }
            else
            {
                hostname_to_ip(broker_ip, optarg);
            }
            break;

        case 'g':
            for (int i = 0; i < gpu_count; i++)
            {
                use_device[i] = false;
            }
            optind--;
            for (; optind < argc && *argv[optind] != '-'; optind++)
            {
                int device = atoi(argv[optind]);
                if (device < 0 || device >= gpu_count) {
                    printf("Invalid gpu index %d\n", device);
                    exit(1);
                }
                use_device[device] = true;
            }
            break;
        default:
            printf("Invalid command %c\n", command);
            exit(1);
        }
    }
    printf("will connect to broker @%s:%d\n", broker_ip, port);

    mining_workers_init(gpu_count);
    setup_gpu_worker_count(gpu_count, gpu_count * parallel_mining_works_per_gpu);

    loop = uv_default_loop();

    uv_tcp_t *socket = (uv_tcp_t *)malloc(sizeof(uv_tcp_t));
    uv_tcp_init(loop, socket);
    uv_tcp_nodelay(socket, 1);
    uv_connect_t *connect = (uv_connect_t *)malloc(sizeof(uv_connect_t));
    struct sockaddr_in dest;
    uv_ip4_addr(broker_ip, port, &dest);
    uv_tcp_connect(connect, socket, (const struct sockaddr *)&dest, on_connect);

    for (int i = 0; i < worker_count; i++)
    {
        uv_async_init(loop, &(mining_workers[i].async), mine_with_async);
        uv_timer_init(loop, &(mining_workers[i].timer));
    }

    uv_timer_t log_timer;
    uv_timer_init(loop, &log_timer);
    uv_timer_start(&log_timer, log_hashrate, 5000, 20000);

    uv_run(loop, UV_RUN_DEFAULT);

    return (0);
}
