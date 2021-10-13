#ifndef ALEPHIUM_BLAKE3_CU
#define ALEPHIUM_BLAKE3_CU

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "constants.h"
#include "messages.h"

#define TRY(x)                                                                                                             \
    {                                                                                                                      \
        cudaGetLastError();                                                                                                \
        x;                                                                                                                 \
        cudaError_t err = cudaGetLastError();                                                                              \
        if (err != cudaSuccess)                                                                                            \
        {                                                                                                                  \
            printf("cudaError %d (%s) calling '%s' (%s line %d)\n", err, cudaGetErrorString(err), #x, __FILE__, __LINE__); \
            exit(1);                                                                                                       \
        }                                                                                                                  \
    }

#define INLINE __inline__

#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024

__constant__ uint32_t IV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL,
                               0xA54FF53AUL, 0x510E527FUL, 0x9B05688CUL,
                               0x1F83D9ABUL, 0x5BE0CD19UL};

__constant__ const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define ROOT (1 << 3)

typedef struct
{
    uint32_t cv[8];
} blake3_chunk_state;

INLINE __device__ void chunk_state_init(blake3_chunk_state *self)
{
    memcpy(self->cv, IV, BLAKE3_KEY_LEN);
}

INLINE __device__ void blake3_compress_in_place(uint32_t cv[8],
                                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                                uint8_t block_len,
                                                uint8_t flags);

INLINE __device__ void chunk_state_update(blake3_chunk_state *self, uint8_t *input, size_t initial_len)
{
    ssize_t input_len = initial_len;
    assert(input_len > 0 && input_len <= BLAKE3_CHUNK_LEN);

    while (input_len > 0)
    {
        ssize_t take = input_len >= BLAKE3_BLOCK_LEN ? BLAKE3_BLOCK_LEN : input_len;

        uint8_t maybe_start_flag = input_len == initial_len ? CHUNK_START : 0;
        input_len -= take;
        uint8_t maybe_end_flag = 0;
        if (input_len == 0)
        {
            maybe_end_flag = CHUNK_END | ROOT;
            memset(input + take, 0, BLAKE3_BLOCK_LEN - take);
        }

        blake3_compress_in_place(self->cv, input, take,
                                 maybe_start_flag | maybe_end_flag);
        input += take;
    }
}

INLINE __device__ uint32_t rotr32(uint32_t w, uint32_t c)
{
    return (w >> c) | (w << (32 - c));
}

INLINE __device__ void g(uint32_t *state, size_t a, size_t b, size_t c, size_t d, uint32_t x, uint32_t y)
{
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

INLINE __device__ void round_fn(uint32_t state[16], const uint32_t *msg, size_t round)
{

    // Select the message schedule based on the round.
    const uint8_t *schedule = MSG_SCHEDULE[round];

    // printf("== state %d: ", round);
    // for (int i = 0; i < 16; i++) {
    //   printf("%d, ", state[i]);
    // }
    // printf("\n");
    // printf("== block %d: ", round);
    // for (int i = 0; i < 16; i++) {
    //   printf("%d, ", msg[schedule[i]]);
    // }
    // printf("\n\n");

    // Mix the columns.
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

    // Mix the rows.
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

INLINE __device__ void compress_pre(uint32_t state[16], const uint32_t cv[8],
                                    const uint8_t block[BLAKE3_BLOCK_LEN],
                                    uint8_t block_len, uint8_t flags)
{
    uint32_t block_words[16];
    memcpy(block_words, block, 16 * 4);

    state[0] = cv[0];
    state[1] = cv[1];
    state[2] = cv[2];
    state[3] = cv[3];
    state[4] = cv[4];
    state[5] = cv[5];
    state[6] = cv[6];
    state[7] = cv[7];
    state[8] = IV[0];
    state[9] = IV[1];
    state[10] = IV[2];
    state[11] = IV[3];
    state[12] = 0;
    state[13] = 0;
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    round_fn(state, &block_words[0], 0);
    round_fn(state, &block_words[0], 1);
    round_fn(state, &block_words[0], 2);
    round_fn(state, &block_words[0], 3);
    round_fn(state, &block_words[0], 4);
    round_fn(state, &block_words[0], 5);
    round_fn(state, &block_words[0], 6);
}

INLINE __device__ void blake3_compress_in_place(uint32_t cv[8],
                                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                                uint8_t block_len,
                                                uint8_t flags)
{
    uint32_t state[16];
    compress_pre(state, cv, block, block_len, flags);
    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];

    // printf("== final state: ");
    // for (int i = 0; i < 16; i++) {
    //   printf("%d, ", state[i]);
    // }
    // printf("\n");
    // printf("== final cv: ");
    // for (int i = 0; i < 16; i++) {
    //   printf("%d, ", cv[i]);
    // }
    // printf("\n\n");
}

typedef struct
{
    blake3_chunk_state chunk;

    uint8_t buf[400];
    size_t buf_len;
    uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

    uint8_t target[32];
    size_t target_len;
    uint32_t from_group;
    uint32_t to_group;

    uint32_t hash_count;
    int found_good_hash;
} blake3_hasher;

INLINE __device__ void blake3_hasher_hash(const blake3_hasher *self, uint8_t *input, size_t input_len, uint8_t *out)
{
    chunk_state_init((blake3_chunk_state *)&self->chunk);
    chunk_state_update((blake3_chunk_state *)&(self->chunk), input, input_len);
    memcpy(out, self->chunk.cv, BLAKE3_OUT_LEN);
}

INLINE __device__ void blake3_hasher_double_hash(blake3_hasher *hasher)
{
    blake3_hasher_hash(hasher, hasher->buf, hasher->buf_len, hasher->hash);
    blake3_hasher_hash(hasher, hasher->hash, 32, hasher->hash);
}

INLINE __device__ bool check_target(uint8_t *hash, uint8_t *target_bytes, size_t target_len)
{
    assert(target_len <= 32);

    ssize_t zero_len = 32 - target_len;
    for (ssize_t i = 0; i < zero_len; i++)
    {
        if (hash[i] != 0)
        {
            return false;
        }
    }
    uint8_t *non_zero_hash = hash + zero_len;
    for (ssize_t i = 0; i < target_len; i++)
    {
        if (non_zero_hash[i] > target_bytes[i])
        {
            return false;
        }
        else if (non_zero_hash[i] < target_bytes[i])
        {
            return true;
        }
    }
    return true;
}

INLINE __device__ bool check_index(uint8_t *hash, uint32_t from_group, uint32_t to_group)
{
    uint8_t big_index = hash[31] % chain_nums;
    return (big_index / group_nums == from_group) && (big_index % group_nums == to_group);
}

INLINE __device__ bool check_hash(uint8_t *hash, uint8_t *target, size_t target_len, uint32_t from_group, uint32_t to_group)
{
    return check_target(hash, target, target_len) && check_index(hash, from_group, to_group);
}

INLINE __device__ void update_nonce(blake3_hasher *hasher, uint64_t delta)
{
    uint64_t *short_nonce = (uint64_t *)hasher->buf;
    *short_nonce += delta;
}

INLINE __device__ void copy_good_nonce(blake3_hasher *thread_hasher, blake3_hasher *global_hasher)
{
    for (int i = 0; i < 24; i++)
    {
        global_hasher->buf[i] = thread_hasher->buf[i];
    }
    for (int i = 0; i < 32; i++)
    {
        global_hasher->hash[i] = thread_hasher->hash[i];
    }
}

__global__ void blake3_hasher_mine(blake3_hasher *global_hasher)
{
    extern __shared__ blake3_hasher s_hashers[];
    int t = threadIdx.x;
    s_hashers[t] = *global_hasher;
    blake3_hasher *hasher = &s_hashers[t];

    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    update_nonce(hasher, tid);

    while (hasher->hash_count < mining_steps)
    {
        hasher->hash_count += 1;

        update_nonce(hasher, stride);
        blake3_hasher_double_hash(hasher);

        if (check_hash(hasher->hash, hasher->target, hasher->target_len, hasher->from_group, hasher->to_group))
        {
            printf("tid %d found it !!\n", tid);
            if (atomicCAS(&global_hasher->found_good_hash, 0, 1) == 0)
            {
                copy_good_nonce(hasher, global_hasher);
            }
            atomicAdd(&global_hasher->hash_count, hasher->hash_count);
            return;
        }
    }
    atomicAdd(&global_hasher->hash_count, hasher->hash_count);
}

#ifdef BLAKE3_TEST
int main()
{
    blob_t blob;
    hex_to_bytes("012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789", &blob);
    blob_t target;
    hex_to_bytes("00004fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", &target);

    print_hex("target string: ", target.blob, target.len);

    blake3_hasher *hasher;
    TRY(cudaMallocManaged(&hasher, sizeof(blake3_hasher)));

    memcpy(hasher->buf, blob.blob, blob.len);
    hasher->buf_len = blob.len;
    memcpy(hasher->target, target.blob, target.len);
    hasher->target_len = target.len;
    hasher->from_group = 2;
    hasher->to_group = 0;

    cudaStream_t stream;
    TRY(cudaStreamCreate(&stream));
    TRY(cudaStreamAttachMemAsync(stream, hasher));

    blake3_hasher_mine<<<1, 16, 16 * sizeof(blake3_hasher), stream>>>(hasher);
    TRY(cudaStreamSynchronize(stream));

    char *hash_string = bytes_to_hex(hasher->hash, 32);
    printf("good: %d\n", hasher->found_good_hash);
    printf("%s\n", hash_string); // 0004ac0418f950947358305af95cd1a81d6277794eb4fb165be18d11895c1170

    memcpy(hasher->buf, blob.blob, blob.len);
    hasher->buf_len = blob.len;
    hasher->buf[0] = 0;
    blake3_hasher_mine<<<1, 1, 1 * sizeof(blake3_hasher), stream>>>(hasher);
    TRY(cudaStreamSynchronize(stream));
    char *hash_string1 = bytes_to_hex(hasher->hash, 32);
    printf("good: %d\n", hasher->found_good_hash);
    printf("%s\n", hash_string1); // 0004ac0418f950947358305af95cd1a81d6277794eb4fb165be18d11895c1170
}
#endif // BLAKE3_TEST

#endif // ALEPHIUM_BLAKE3_CU