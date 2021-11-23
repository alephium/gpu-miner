#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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

#define INLINE static inline

#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_BUF_CAP 384
#define BLAKE3_BUF_LEN 326

#define IV_0 0x6A09E667UL
#define IV_1 0xBB67AE85UL
#define IV_2 0x3C6EF372UL
#define IV_3 0xA54FF53AUL
#define IV_4 0x510E527FUL
#define IV_5 0x9B05688CUL
#define IV_6 0x1F83D9ABUL
#define IV_7 0x5BE0CD19UL

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define ROOT (1 << 3)

INLINE void cv_state_init(uint32_t *cv)
{
    cv[0] = IV_0;
    cv[1] = IV_1;
    cv[2] = IV_2;
    cv[3] = IV_3;
    cv[4] = IV_4;
    cv[5] = IV_5;
    cv[6] = IV_6;
    cv[7] = IV_7;
}

INLINE void blake3_compress_in_place(uint32_t cv[8],
                                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                                uint8_t block_len,
                                                uint8_t flags);

INLINE void chunk_state_update(uint32_t cv[8], uint8_t *input, size_t initial_len)
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

        blake3_compress_in_place(cv, input, take, maybe_start_flag | maybe_end_flag);
        input += take;
    }
}

INLINE uint32_t rotr32(uint32_t w, uint32_t c)
{
    return (w >> c) | (w << (32 - c));
}

#define G(a, b, c, d, x, y)                         \
    do                                              \
    {                                               \
        state[a] = state[a] + state[b] + x;         \
        state[d] = rotr32(state[d] ^ state[a], 16); \
        state[c] = state[c] + state[d];             \
        state[b] = rotr32(state[b] ^ state[c], 12); \
        state[a] = state[a] + state[b] + y;         \
        state[d] = rotr32(state[d] ^ state[a], 8);  \
        state[c] = state[c] + state[d];             \
        state[b] = rotr32(state[b] ^ state[c], 7);  \
    } while (0)

#define Z00 0
#define Z01 1
#define Z02 2
#define Z03 3
#define Z04 4
#define Z05 5
#define Z06 6
#define Z07 7
#define Z08 8
#define Z09 9
#define Z0A 10
#define Z0B 11
#define Z0C 12
#define Z0D 13
#define Z0E 14
#define Z0F 15
#define Z10 2
#define Z11 6
#define Z12 3
#define Z13 10
#define Z14 7
#define Z15 0
#define Z16 4
#define Z17 13
#define Z18 1
#define Z19 11
#define Z1A 12
#define Z1B 5
#define Z1C 9
#define Z1D 14
#define Z1E 15
#define Z1F 8
#define Z20 3
#define Z21 4
#define Z22 10
#define Z23 12
#define Z24 13
#define Z25 2
#define Z26 7
#define Z27 14
#define Z28 6
#define Z29 5
#define Z2A 9
#define Z2B 0
#define Z2C 11
#define Z2D 15
#define Z2E 8
#define Z2F 1
#define Z30 10
#define Z31 7
#define Z32 12
#define Z33 9
#define Z34 14
#define Z35 3
#define Z36 13
#define Z37 15
#define Z38 4
#define Z39 0
#define Z3A 11
#define Z3B 2
#define Z3C 5
#define Z3D 8
#define Z3E 1
#define Z3F 6
#define Z40 12
#define Z41 13
#define Z42 9
#define Z43 11
#define Z44 15
#define Z45 10
#define Z46 14
#define Z47 8
#define Z48 7
#define Z49 2
#define Z4A 5
#define Z4B 3
#define Z4C 0
#define Z4D 1
#define Z4E 6
#define Z4F 4
#define Z50 9
#define Z51 14
#define Z52 11
#define Z53 5
#define Z54 8
#define Z55 12
#define Z56 15
#define Z57 1
#define Z58 13
#define Z59 3
#define Z5A 0
#define Z5B 10
#define Z5C 2
#define Z5D 6
#define Z5E 4
#define Z5F 7
#define Z60 11
#define Z61 15
#define Z62 5
#define Z63 0
#define Z64 1
#define Z65 9
#define Z66 8
#define Z67 6
#define Z68 14
#define Z69 10
#define Z6A 2
#define Z6B 12
#define Z6C 3
#define Z6D 4
#define Z6E 7
#define Z6F 13

#define Mx(r, i) (block_words[Z##r##i])

#define ROUND_S(r)                                 \
    do                                             \
    {                                              \
        G(0x0, 0x4, 0x8, 0xC, Mx(r, 0), Mx(r, 1)); \
        G(0x1, 0x5, 0x9, 0xD, Mx(r, 2), Mx(r, 3)); \
        G(0x2, 0x6, 0xA, 0xE, Mx(r, 4), Mx(r, 5)); \
        G(0x3, 0x7, 0xB, 0xF, Mx(r, 6), Mx(r, 7)); \
        G(0x0, 0x5, 0xA, 0xF, Mx(r, 8), Mx(r, 9)); \
        G(0x1, 0x6, 0xB, 0xC, Mx(r, A), Mx(r, B)); \
        G(0x2, 0x7, 0x8, 0xD, Mx(r, C), Mx(r, D)); \
        G(0x3, 0x4, 0x9, 0xE, Mx(r, E), Mx(r, F)); \
    } while (0)

INLINE void compress_pre(uint32_t state[16], const uint32_t cv[8],
                                    const uint8_t block[BLAKE3_BLOCK_LEN],
                                    uint8_t block_len, uint8_t flags)
{
    uint32_t *block_words = (uint32_t *)block;

    state[0] = cv[0];
    state[1] = cv[1];
    state[2] = cv[2];
    state[3] = cv[3];
    state[4] = cv[4];
    state[5] = cv[5];
    state[6] = cv[6];
    state[7] = cv[7];
    state[8] = IV_0;
    state[9] = IV_1;
    state[10] = IV_2;
    state[11] = IV_3;
    state[12] = 0;
    state[13] = 0;
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    ROUND_S(0);
    ROUND_S(1);
    ROUND_S(2);
    ROUND_S(3);
    ROUND_S(4);
    ROUND_S(5);
    ROUND_S(6);
}

INLINE void blake3_compress_in_place(uint32_t cv[8],
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
    uint8_t buf[BLAKE3_BUF_CAP];

    uint32_t cv[8];

    uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

    uint8_t target[32];
    uint32_t from_group;
    uint32_t to_group;

    uint32_t hash_count;
    int found_good_hash;
} blake3_hasher;

INLINE void blake3_hasher_hash(const blake3_hasher *self, uint8_t *input, size_t input_len, uint8_t *out)
{
    cv_state_init((uint32_t *)self->cv);
    chunk_state_update((uint32_t *)&self->cv, input, input_len);
#pragma unroll
    for (int i = 0; i < 8; i ++) {
        ((uint32_t *) out)[i] = self->cv[i];
    }
}

INLINE void blake3_hasher_double_hash(blake3_hasher *hasher)
{
    blake3_hasher_hash(hasher, hasher->buf, BLAKE3_BUF_LEN, hasher->hash);
    blake3_hasher_hash(hasher, hasher->hash, BLAKE3_OUT_LEN, hasher->hash);
}

INLINE bool check_target(uint8_t *hash, uint8_t *target)
{
#pragma unroll
    for (ssize_t i = 0; i < 32; i++)
    {
        if (hash[i] > target[i])
        {
            return false;
        }
        else if (hash[i] < target[i])
        {
            return true;
        }
    }
    return true;
}

INLINE bool check_index(uint8_t *hash, uint32_t from_group, uint32_t to_group)
{
    uint8_t big_index = hash[31] % chain_nums;
    return (big_index / group_nums == from_group) && (big_index % group_nums == to_group);
}

INLINE bool check_hash(uint8_t *hash, uint8_t *target, uint32_t from_group, uint32_t to_group)
{
    return check_target(hash, target) && check_index(hash, from_group, to_group);
}

INLINE void update_nonce(blake3_hasher *hasher, uint64_t delta)
{
    uint64_t *short_nonce = (uint64_t *)hasher->buf;
    *short_nonce += delta;
}

INLINE void copy_good_nonce(blake3_hasher *thread_hasher, blake3_hasher *global_hasher)
{
#pragma unroll
    for (int i = 0; i < 24; i++)
    {
        global_hasher->buf[i] = thread_hasher->buf[i];
    }
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        global_hasher->hash[i] = thread_hasher->hash[i];
    }
}

__kernel void blake3_hasher_mine(__global blake3_hasher *global_hasher)
{
    blake3_hasher local_hasher = *global_hasher;
    blake3_hasher *hasher = &local_hasher;

    hasher->hash_count = 0;

    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t *short_nonce = (uint64_t *)hasher->buf;
    *short_nonce = (*short_nonce) / stride * stride + tid;

    while (hasher->hash_count < mining_steps)
    {
        hasher->hash_count += 1;

        *short_nonce += stride;
        blake3_hasher_double_hash(hasher);

        if (check_hash(hasher->hash, hasher->target, hasher->from_group, hasher->to_group))
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
