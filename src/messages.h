#ifndef ALEPHIUM_MESSAGE_H
#define ALEPHIUM_MESSAGE_H

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

typedef struct blob_t {
    uint8_t *blob;
    ssize_t len;
} blob_t;

void free_blob(blob_t *blob)
{
    free(blob->blob);
}

char *bytes_to_hex(uint8_t *bytes, ssize_t len)
{
    ssize_t hex_len = 2 * len + 1;
    char *hex_string = (char *)malloc(hex_len);
    memset(hex_string, 0, hex_len);

    uint8_t *byte_cursor = bytes;
    char *hex_cursor = hex_string;
    ssize_t count = 0;
    while (count < len) {
        sprintf(hex_cursor, "%02x", *byte_cursor);
        byte_cursor++;
        count++;
        hex_cursor += 2;
    }

    return hex_string;
}

void print_hex(const char* prefix, uint8_t *data, ssize_t nread)
{
    char *hex_string = bytes_to_hex(data, nread);
    printf("%s: %s\n", prefix, hex_string);
    free(hex_string);
}

char hex_to_byte(char hex)
{
    if (hex >= '0' && hex <= '9') {
        return hex - '0';
    } else if (hex >= 'a' && hex <= 'f') {
        return hex - 'a' + 10;
    } else {
        exit(1);
    }
}

void hex_to_bytes(const char *hex_data, blob_t *buf)
{
    size_t hex_len = strlen(hex_data);
    assert(hex_len % 2 == 0);

    buf->len = hex_len / 2;
    buf->blob = (uint8_t *)malloc(buf->len);
    memset(buf->blob, 0, buf->len);

    for (size_t pos = 0; pos < hex_len; pos += 2) {
        char left = hex_to_byte(hex_data[pos]);
        char right = hex_to_byte(hex_data[pos + 1]);
        buf->blob[pos / 2] = (left << 4) + right;
    }
}

typedef struct job_t {
    int from_group;
    int to_group;
    blob_t header_blob;
    blob_t txs_blob;
    blob_t target;
} job_t;

void free_job(job_t *job) {
    free_blob(&job->header_blob);
    free_blob(&job->txs_blob);
    free_blob(&job->target);
    free(job);
}

typedef struct jobs_t {
    job_t **jobs;
    size_t len;
} jobs_t;

void free_jobs(jobs_t *jobs)
{
    for (size_t i = 0; i < jobs->len; i++) {
        free_job(jobs->jobs[i]);
    }
    free(jobs->jobs);
}

typedef struct submit_result_t {
    int from_group;
    int to_group;
    bool status;
} submit_result_t;

typedef enum server_message_kind {
    JOBS,
    SUBMIT_RESULT,
} server_message_kind;

typedef struct server_message_t {
    server_message_kind kind;
    union {
        jobs_t *jobs;
        submit_result_t *submit_result;
    };
} server_message_t;

void free_server_message_except_jobs(server_message_t *message)
{
    switch (message->kind)
    {
    case JOBS:
        free(message->jobs->jobs);
        free(message->jobs);
        break;

    case SUBMIT_RESULT:
        free(message->submit_result);
        break;
    }

    free(message);
}

void write_size(uint8_t **bytes, ssize_t size)
{
    (*bytes)[0] = (size >> 24) & 0xFF;
    (*bytes)[1] = (size >> 16) & 0xFF;
    (*bytes)[2] = (size >> 8) & 0xFF;
    (*bytes)[3] = size & 0xFF;
    *bytes = *bytes + 4;
    return;
}

ssize_t decode_size(uint8_t *bytes)
{
    return bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
}

ssize_t extract_size(uint8_t **bytes)
{
    ssize_t size = decode_size(*bytes);
    *bytes = *bytes + 4;
    return size;
}

void write_byte(uint8_t **bytes, uint8_t byte)
{
    (*bytes)[0] = byte;
    *bytes = *bytes + 1;
}

uint8_t extract_byte(uint8_t **bytes)
{
    uint8_t byte = **bytes;
    *bytes = *bytes + 1;
    return byte;
}

bool extract_bool(uint8_t **bytes)
{
    uint8_t byte = extract_byte(bytes);
    switch (byte)
    {
    case 0:
        return false;
    case 1:
        return true;
    default:
        fprintf(stderr, "Invaid bool value");
        exit(1);
    }
}

void write_bytes(uint8_t **bytes, uint8_t *data, ssize_t len)
{
    memcpy(*bytes, data, len);
    *bytes = *bytes + len;
}

void write_blob(uint8_t **bytes, blob_t *blob)
{
    write_bytes(bytes, blob->blob, blob->len);
}

void extract_blob(uint8_t **bytes, blob_t *blob)
{
    ssize_t size = extract_size(bytes);
    blob->len = size;
    blob->blob = (uint8_t *)malloc(size * sizeof(uint8_t));
    memcpy(blob->blob, *bytes, size);
    *bytes = *bytes + size;

    // printf("blob: %ld\n", blob->len);
    // printf("%s\n", bytes_to_hex(blob->blob, blob->len));
}

void extract_job(uint8_t **bytes, job_t *job)
{
    job->from_group = extract_size(bytes);
    job->to_group = extract_size(bytes);
    // printf("group: %d, %d\n", job->from_group, job->to_group);
    extract_blob(bytes, &job->header_blob);
    extract_blob(bytes, &job->txs_blob);
    extract_blob(bytes, &job->target);
}

void extract_jobs(uint8_t **bytes, jobs_t *jobs)
{
    ssize_t jobs_size = extract_size(bytes);

    // printf("jobs: %ld\n", jobs_size);

    jobs->len = jobs_size;
    jobs->jobs = (job_t **)malloc(jobs_size * sizeof(job_t*));
    for(ssize_t i = 0; i < jobs_size; i++) {
        jobs->jobs[i] = (job_t *)malloc(sizeof(job_t));
        extract_job(bytes, (jobs->jobs[i]));
    }
}

void extract_submit_result(uint8_t **bytes, submit_result_t *result)
{
    result->from_group = extract_size(bytes);
    result->to_group = extract_size(bytes);
    result->status = extract_bool(bytes);
}

server_message_t *decode_server_message(blob_t *blob)
{
    uint8_t *bytes = blob->blob;
    ssize_t len = blob->len;

    if (len <= 4) {
        return NULL; // not enough bytes for decoding
    }

    uint8_t *pos = bytes;
    ssize_t message_size = extract_size(&pos);
    assert(pos == bytes + 4);

    ssize_t message_byte_size = message_size + 4;
    if (len < message_byte_size) {
        return NULL; // not enough bytes for decoding
    }

    server_message_t *server_message = (server_message_t *)malloc(sizeof(server_message_t));
    switch (extract_byte(&pos))
    {
    case 0:
        server_message->kind = JOBS;
        server_message->jobs = (jobs_t *)malloc(sizeof(jobs_t));
        extract_jobs(&pos, server_message->jobs);

        // printf("%p, %p, %p\n", bytes, pos, bytes + len);
        break;

    case 1:
        server_message->kind = SUBMIT_RESULT;
        server_message->submit_result = (submit_result_t *)malloc(sizeof(submit_result_t));
        extract_submit_result(&pos, server_message->submit_result);
        break;

    default:
        fprintf(stderr, "Invalid server message kind");
        exit(1);
    }

    assert(pos == (bytes + message_byte_size));
    if (message_byte_size < len) {
        blob->len = len - message_byte_size;
        memmove(blob->blob, pos, blob->len);
    } else {
        blob->len = 0;
    }

    return server_message;
}

#endif // ALEPHIUM_MESSAGE_H
