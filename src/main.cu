#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "constants.h"
#include "uv.h"
#include "messages.h"
#include "blake3.cu"
#include "pow.h"
#include "worker.h"
#include "template.h"
#include "mining.h"
#include "cuda_helper.cu"

uv_loop_t *loop;
uv_stream_t *tcp;

time_point_t start_time = Time::now();

void on_write_end(uv_write_t *req, int status)
{
    if (status == -1) {
        fprintf(stderr, "error on_write_end");
        exit(1);
    }
    free(req);
    printf("sent new block\n");
}

void submit_new_block(mining_worker_t *worker)
{
    if (!expire_template_for_new_block(load_worker__template(worker))) {
        printf("mined a parallel block, will not submit\n");
        return;
    }

    ssize_t buf_size = write_new_block(worker);
    uv_buf_t buf = uv_buf_init((char *)write_buffers[worker->id], buf_size);
    print_hex("new block", (uint8_t *)worker->hasher->hash, 32);

    uv_write_t *write_req = (uv_write_t *)malloc(sizeof(uv_write_t));
    uint32_t buf_count = 1;

    printf("sending new block\n");
    uv_write(write_req, tcp, &buf, buf_count, on_write_end);
}

void mine(uv_work_t *req)
{
    int32_t to_mine_index = next_chain_to_mine();
    if (to_mine_index == -1) {
        printf("No task available, quitting...\n");
        exit(1);
    }

    mining_worker_t *worker = load_req_worker(req);
    mining_counts[to_mine_index].fetch_add(mining_steps);
    setup_template(worker, load_template(to_mine_index));

    start_worker_mining(worker);
}

void after_mine(uv_work_t *req, int status)
{
    mining_worker_t *worker = load_req_worker(req);
    // printf("after mine: %d %d %d\n", work->job->from_group, work->job->to_group, worker->hash_count);

    if (load_worker__found_good_hash(worker)) {
        submit_new_block(worker);
    }

    mining_template_t *template_ptr = load_worker__template(worker);
    job_t *job = template_ptr->job;
    uint32_t chain_index = job->from_group * group_nums + job->to_group;
    mining_counts[chain_index].fetch_sub(mining_steps);
    mining_counts[chain_index].fetch_add(worker->hasher->hash_count);

    free_template(template_ptr);
    uv_queue_work(loop, req, mine, after_mine);
}

void start_mining()
{
    assert(mining_templates_initialized == true);

    start_time = Time::now();

    for (uint32_t i = 0; i < parallel_mining_works; i++) {
        uv_queue_work(loop, &req[i], mine, after_mine);
    }
}

void start_mining_if_needed()
{
    if (!mining_templates_initialized) {
        bool all_initialized = true;
        for (int i = 0; i < chain_nums; i++) {
            if (load_template(i) == NULL) {
                all_initialized = false;
                break;
            }
        }
        if (all_initialized) {
            mining_templates_initialized = true;
            start_mining();
        }
    }
}

void alloc_buffer(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf)
{
    buf->base = (char *)malloc(suggested_size);
    buf->len = suggested_size;
}

uint8_t read_buf[2048 * 1024 * chain_nums];
blob_t read_blob = { read_buf, 0 };
server_message_t *decode_buf(const uv_buf_t *buf, ssize_t nread)
{
    if (read_blob.len == 0) {
        read_blob.blob = (uint8_t *)buf->base;
        read_blob.len = nread;
        server_message_t *message = decode_server_message(&read_blob);
        if (message) {
            // some bytes left
            if (read_blob.len > 0) {
                memcpy(read_buf, read_blob.blob, read_blob.len);
                read_blob.blob = read_buf;
            }
            return message;
        } else { // no bytes consumed
            memcpy(read_buf, buf->base, nread);
            read_blob.blob = read_buf;
            read_blob.len = nread;
            return NULL;
        }
    } else {
        assert(read_blob.blob == read_buf);
        memcpy(read_buf + read_blob.len, buf->base, nread);
        read_blob.len += nread;
        return decode_server_message(&read_blob);
    }
}

void on_read(uv_stream_t *server, ssize_t nread, const uv_buf_t *buf)
{
    time_point_t current_time = Time::now();
    if (current_time > start_time) {

        uint64_t total_hash = 0;
        for (int i = 0; i < chain_nums; i++) {
            total_hash += mining_counts[i].load();
        }
        duration_t eplased = current_time - start_time;
        printf("hashrate: %f (hash/sec)\n", total_hash / eplased.count());
    }

    if (nread < 0) {
        fprintf(stderr, "error on_read %ld\n", nread);
        exit(1);
    }

    if (nread == 0) {
        return;
    }

    server_message_t *message = decode_buf(buf, nread);
    if (!message) {
        return;
    }

    printf("message type: %d\n", message->kind);
    switch (message->kind)
    {
    case JOBS:
        for (int i = 0; i < message->jobs->len; i ++) {
            update_templates(message->jobs->jobs[i]);
        }
        start_mining_if_needed();
        break;

    case SUBMIT_RESULT:
        printf("submitted: %d -> %d: %d \n", message->submit_result->from_group, message->submit_result->to_group, message->submit_result->status);
        break;
    }

    free(buf->base);
    free_server_message_except_jobs(message);
    // uv_close((uv_handle_t *) server, free_close_cb);
}

void on_connect(uv_connect_t *req, int status)
{
    if (status == -1) {
        fprintf(stderr, "connection error");
        return;
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
    if (res != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
      return 1;
    }

    struct sockaddr_in *h = (struct sockaddr_in *) servinfo->ai_addr;
    strcpy(ip_address, inet_ntoa(h->sin_addr));

    freeaddrinfo(servinfo);
    return 0;
}

int main(int argc, char **argv)
{
    int grid_size;
    int block_size = 32;
    config_cuda(&grid_size, &block_size);
    mining_workers_init(grid_size, block_size);
    printf("Cuda grid size: %d, block size: %d\n", grid_size, block_size);

    char broker_ip[16];
    memset(broker_ip, '\0', sizeof(broker_ip));

    if (argc >= 2) {
      if (is_valid_ip_address(argv[1])) {
        strcpy(broker_ip, argv[1]);
      } else {
        hostname_to_ip(broker_ip, argv[1]);
      }
    } else {
      strcpy(broker_ip, "127.0.0.1");
    }

    printf("Will connect to broker @%s:10973\n", broker_ip);

    loop = uv_default_loop();

    uv_tcp_t* socket = (uv_tcp_t *)malloc(sizeof(uv_tcp_t));
    uv_tcp_init(loop, socket);

    uv_connect_t* connect = (uv_connect_t *)malloc(sizeof(uv_connect_t));

    struct sockaddr_in dest;
    uv_ip4_addr(broker_ip, 10973, &dest);

    uv_tcp_connect(connect, socket, (const struct sockaddr*)&dest, on_connect);
    uv_run(loop, UV_RUN_DEFAULT);

    return (0);
}
