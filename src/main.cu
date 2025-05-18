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
#include "log.h"

#define MAX_MESSAGE_SIZE (2048 * 1024)
#define BUFFER_SIZE (MAX_MESSAGE_SIZE * chain_nums)

typedef struct {
    uv_mutex_t lock;
    uint8_t buffer[BUFFER_SIZE];
    ssize_t length;
} message_buffer_t;

message_buffer_t message_buffer;
uv_async_t message_handler;

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

int port = 10973;
char broker_ip[16];
uv_timer_t reconnect_timer;
uv_tcp_t *uv_socket;
uv_connect_t *uv_connect;

void setup_gpu_worker_count(int _gpu_count, int _worker_count)
{
    gpu_count.store(_gpu_count);
    worker_count.store(_worker_count);
}

void on_write_end(uv_write_t *req, int status)
{
    if (status < 0)
    {
        LOGERR("error on_write_end %d\n", status);
    }
    free(req);
}

void print_block(job_t *job, uint8_t *hash)
{
    char *hex_string = bytes_to_hex(hash, 32);
    LOG("new block: %d -> %d, height: %d, hash: %s\n", job->from_group, job->to_group, job->height, hex_string);
    free(hex_string);
}

std::mutex write_mutex;
uint8_t write_buffer[4096 * 1024];
void submit_new_block(mining_worker_t *worker)
{
    mining_template_t *template_ptr = load_worker__template(worker);
    print_block(template_ptr->job, (uint8_t *) hasher_hash(worker, true));
    expire_template_for_new_block(template_ptr);

    const std::lock_guard<std::mutex> lock(write_mutex);

    ssize_t buf_size = write_new_block(worker, write_buffer);
    uv_buf_t buf = uv_buf_init((char *)write_buffer, buf_size);

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
        LOG("waiting for new tasks\n");
        worker->timer.data = worker;
        uv_timer_start(&worker->timer, mine_with_timer, 500, 0);
    } else {
        mining_counts[to_mine_index].fetch_add(mining_steps);
        setup_template(worker, load_template(to_mine_index));

        start_worker_mining(worker);

        duration_t elapsed = Time::now() - start;
        // LOG("=== mining time: %fs\n", elapsed.count());
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
        LOG("hashrate: %.0f MH/s ", total_mining_count.load() / eplased.count() / 1000000);
        for (int i = 0; i < gpu_count; i++)
        {
            LOG_WITHOUT_TS("gpu%d: %.0f MH/s ", i, device_mining_count[i].load() / eplased.count() / 1000000);
        }
        LOG_WITHOUT_TS("solutions: %u\n", found_solutions.load(std::memory_order_relaxed));
    }
}

void connect_to_broker();

void try_to_reconnect(uv_timer_t *timer){
    message_buffer.length = 0;
    free(uv_socket);
    free(uv_connect);
    connect_to_broker();
    uv_timer_stop(timer);
}

void on_read(uv_stream_t *server, ssize_t nread, const uv_buf_t *buf)
{
    if (nread < 0)
    {
        LOGERR("error on_read %ld: might be that the full node is not synced, or miner wallets are not setup, try to reconnect\n", nread);
        uv_timer_start(&reconnect_timer, try_to_reconnect, 5000, 0);
        return;
    }

    if (nread == 0)
    {
        return;
    }

    uv_mutex_lock(&message_buffer.lock);
    assert(message_buffer.length + nread <= BUFFER_SIZE);
    memcpy(message_buffer.buffer + message_buffer.length, buf->base, nread);
    message_buffer.length += nread;
    uv_mutex_unlock(&message_buffer.lock);

    uv_async_send(&message_handler);
    free(buf->base);
}

uint8_t latest_job[MAX_MESSAGE_SIZE];

void process_message(uv_async_t* handle) {
    bool found_latest_job = false;
    ssize_t latest_job_offset = 0;
    ssize_t latest_job_len = 0;
    ssize_t offset = 0;

    uv_mutex_lock(&message_buffer.lock);
    uint8_t *buf = message_buffer.buffer;
    ssize_t len = message_buffer.length;

    // try to read the latest jobs message from the buffer.
    while (len - offset >= 4)
    {
        ssize_t message_size = decode_size(buf + offset);
        ssize_t total_message_size = 4 + message_size;
        // ignore the submit result message for simplicity
        if (total_message_size == 47)
        {
            offset += total_message_size;
            continue;
        }
        if (len - offset >= total_message_size)
        {
            latest_job_offset = offset + 4;
            latest_job_len = message_size;
            offset += total_message_size;
            found_latest_job = true;
        } else
        {
            break;
        }
    }

    if (found_latest_job)
    {
        memcpy(latest_job, buf + latest_job_offset, latest_job_len);
    }

    if (offset > 0)
    {
        if (offset == len)
        {
            message_buffer.length = 0;
        }
        else
        {
            ssize_t remain = len - offset;
            memmove(message_buffer.buffer, buf + offset, remain);
            message_buffer.length = remain;
        }
    }
    uv_mutex_unlock(&message_buffer.lock);

    if (found_latest_job)
    {
        server_message_t *message = decode_server_message(latest_job, latest_job_len);
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
                char *block_hash_hex = bytes_to_hex(message->submit_result->block_hash, 32);
                LOG(
                    "submitted: %d -> %d, %s: %d \n",
                    message->submit_result->from_group,
                    message->submit_result->to_group,
                    block_hash_hex,
                    message->submit_result->status
                );
                free(block_hash_hex);
                break;
            }
            free_server_message_except_jobs(message);
        }
    }
}

void on_connect(uv_connect_t *req, int status)
{
    if (status < 0)
    {
        LOGERR("connection error %d: might be that the full node is not reachable, try to reconnect\n", status);
        uv_timer_start(&reconnect_timer, try_to_reconnect, 5000, 0);
        return;
    }
    LOG("the server is connected %d %p\n", status, req);

    tcp = req->handle;
    uv_read_start(req->handle, alloc_buffer, on_read);
}

void connect_to_broker(){
    uv_socket = (uv_tcp_t *)malloc(sizeof(uv_tcp_t));
    uv_tcp_init(loop, uv_socket);
    uv_tcp_nodelay(uv_socket, 1);
    uv_connect = (uv_connect_t *)malloc(sizeof(uv_connect_t));
    struct sockaddr_in dest;
    uv_ip4_addr(broker_ip, port, &dest);
    uv_tcp_connect(uv_connect, uv_socket, (const struct sockaddr *)&dest, on_connect);
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
        LOGERR("getaddrinfo: %s\n", gai_strerror(res));
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
        LOGERR("Initialize winsock failed: %d\n", rc);
        exit(1);
    }
    #endif

    LOG("Running gpu-miner version : %s\n", MINER_VERSION);

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    LOG("GPU count: %d\n", gpu_count);
    for (int i = 0; i < gpu_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        LOG("GPU #%d - %s has #%d cores\n", i, prop.name, get_device_cores(i));
        use_device[i] = true;
    }

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
                    LOGERR("Invalid gpu index %d\n", device);
                    exit(1);
                }
                use_device[device] = true;
            }
            break;
        default:
            LOGERR("Invalid command %c\n", command);
            exit(1);
        }
    }
    LOG("will connect to broker @%s:%d\n", broker_ip, port);

    #ifdef __linux__
    signal(SIGPIPE, SIG_IGN);
    #endif

    mining_workers_init(gpu_count);
    setup_gpu_worker_count(gpu_count, gpu_count * parallel_mining_works_per_gpu);

    loop = uv_default_loop();
    uv_mutex_init(&message_buffer.lock);
    message_buffer.length = 0;
    uv_async_init(loop, &message_handler, process_message);
    uv_timer_init(loop, &reconnect_timer);
    connect_to_broker();

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
