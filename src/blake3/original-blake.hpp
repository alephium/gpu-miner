#ifndef ALEPHIUM_ORIGINAL_BLAKE_H
#define ALEPHIUM_ORIGINAL_BLAKE_H

#include "blake3-common.hpp"

namespace ref_blake{

#define REF_Z00 0
#define REF_Z01 1
#define REF_Z02 2
#define REF_Z03 3
#define REF_Z04 4
#define REF_Z05 5
#define REF_Z06 6
#define REF_Z07 7
#define REF_Z08 8
#define REF_Z09 9
#define REF_Z0A 10
#define REF_Z0B 11
#define REF_Z0C 12
#define REF_Z0D 13
#define REF_Z0E 14
#define REF_Z0F 15
#define REF_Z10 2
#define REF_Z11 6
#define REF_Z12 3
#define REF_Z13 10
#define REF_Z14 7
#define REF_Z15 0
#define REF_Z16 4
#define REF_Z17 13
#define REF_Z18 1
#define REF_Z19 11
#define REF_Z1A 12
#define REF_Z1B 5
#define REF_Z1C 9
#define REF_Z1D 14
#define REF_Z1E 15
#define REF_Z1F 8
#define REF_Z20 3
#define REF_Z21 4
#define REF_Z22 10
#define REF_Z23 12
#define REF_Z24 13
#define REF_Z25 2
#define REF_Z26 7
#define REF_Z27 14
#define REF_Z28 6
#define REF_Z29 5
#define REF_Z2A 9
#define REF_Z2B 0
#define REF_Z2C 11
#define REF_Z2D 15
#define REF_Z2E 8
#define REF_Z2F 1
#define REF_Z30 10
#define REF_Z31 7
#define REF_Z32 12
#define REF_Z33 9
#define REF_Z34 14
#define REF_Z35 3
#define REF_Z36 13
#define REF_Z37 15
#define REF_Z38 4
#define REF_Z39 0
#define REF_Z3A 11
#define REF_Z3B 2
#define REF_Z3C 5
#define REF_Z3D 8
#define REF_Z3E 1
#define REF_Z3F 6
#define REF_Z40 12
#define REF_Z41 13
#define REF_Z42 9
#define REF_Z43 11
#define REF_Z44 15
#define REF_Z45 10
#define REF_Z46 14
#define REF_Z47 8
#define REF_Z48 7
#define REF_Z49 2
#define REF_Z4A 5
#define REF_Z4B 3
#define REF_Z4C 0
#define REF_Z4D 1
#define REF_Z4E 6
#define REF_Z4F 4
#define REF_Z50 9
#define REF_Z51 14
#define REF_Z52 11
#define REF_Z53 5
#define REF_Z54 8
#define REF_Z55 12
#define REF_Z56 15
#define REF_Z57 1
#define REF_Z58 13
#define REF_Z59 3
#define REF_Z5A 0
#define REF_Z5B 10
#define REF_Z5C 2
#define REF_Z5D 6
#define REF_Z5E 4
#define REF_Z5F 7
#define REF_Z60 11
#define REF_Z61 15
#define REF_Z62 5
#define REF_Z63 0
#define REF_Z64 1
#define REF_Z65 9
#define REF_Z66 8
#define REF_Z67 6
#define REF_Z68 14
#define REF_Z69 10
#define REF_Z6A 2
#define REF_Z6B 12
#define REF_Z6C 3
#define REF_Z6D 4
#define REF_Z6E 7
#define REF_Z6F 13

INLINE __device__ void cv_state_init(uint32_t *cv)
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

INLINE __device__ void blake3_compress_in_place(uint32_t cv[8],
                                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                                uint8_t block_len,
                                                uint8_t flags);

INLINE __device__ void chunk_state_update(uint32_t cv[8], uint8_t *input, size_t initial_len)
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

INLINE __device__ uint32_t rotr32(uint32_t w, uint32_t c)
{
    return (w >> c) | (w << (32 - c));
}

#define REF_G(a, b, c, d, x, y)                         \
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

#define REF_Mx(r, i) (block_words[REF_Z##r##i])

#define ROUND_S(r)                                 \
    do                                             \
    {                                              \
        REF_G(0x0, 0x4, 0x8, 0xC, REF_Mx(r, 0), REF_Mx(r, 1)); \
        REF_G(0x1, 0x5, 0x9, 0xD, REF_Mx(r, 2), REF_Mx(r, 3)); \
        REF_G(0x2, 0x6, 0xA, 0xE, REF_Mx(r, 4), REF_Mx(r, 5)); \
        REF_G(0x3, 0x7, 0xB, 0xF, REF_Mx(r, 6), REF_Mx(r, 7)); \
        REF_G(0x0, 0x5, 0xA, 0xF, REF_Mx(r, 8), REF_Mx(r, 9)); \
        REF_G(0x1, 0x6, 0xB, 0xC, REF_Mx(r, A), REF_Mx(r, B)); \
        REF_G(0x2, 0x7, 0x8, 0xD, REF_Mx(r, C), REF_Mx(r, D)); \
        REF_G(0x3, 0x4, 0x9, 0xE, REF_Mx(r, E), REF_Mx(r, F)); \
    } while (0)

INLINE __device__ void compress_pre(uint32_t state[16], const uint32_t cv[8],
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
    uint8_t buf[BLAKE3_BUF_CAP];

    uint32_t cv[8];

    uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

    uint8_t target[32];
    uint32_t from_group;
    uint32_t to_group;

    uint32_t hash_count;
    int found_good_hash;
} blake3_hasher;

INLINE __device__ void blake3_hasher_hash(const blake3_hasher *self, uint8_t *input, size_t input_len, uint8_t *out)
{
    cv_state_init((uint32_t *)self->cv);
    chunk_state_update((uint32_t *)&self->cv, input, input_len);
#pragma unroll
    for (int i = 0; i < 8; i ++) {
        ((uint32_t *) out)[i] = self->cv[i];
    }
}

INLINE __device__ void blake3_hasher_double_hash(blake3_hasher *hasher)
{
    blake3_hasher_hash(hasher, hasher->buf, BLAKE3_BUF_LEN, hasher->hash);
    blake3_hasher_hash(hasher, hasher->hash, BLAKE3_OUT_LEN, hasher->hash);
}

INLINE __device__ bool check_target(uint8_t *hash, uint8_t *target)
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

INLINE __device__ bool check_index(uint8_t *hash, uint32_t from_group, uint32_t to_group)
{
    uint8_t big_index = hash[31] % chain_nums;
    return (big_index / group_nums == from_group) && (big_index % group_nums == to_group);
}

INLINE __device__ bool check_hash(uint8_t *hash, uint8_t *target, uint32_t from_group, uint32_t to_group)
{
    return check_target(hash, target) && check_index(hash, from_group, to_group);
}

INLINE __device__ void update_nonce(blake3_hasher *hasher, uint64_t delta)
{
    uint64_t *short_nonce = (uint64_t *)hasher->buf;
    *short_nonce += delta;
}

INLINE __device__ void copy_good_nonce(blake3_hasher *thread_hasher, blake3_hasher *global_hasher)
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

__global__ void blake3_hasher_mine(void *global_hasher)
{
    blake3_hasher local_hasher = *reinterpret_cast<blake3_hasher*>(global_hasher);
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
            if (atomicCAS(&reinterpret_cast<blake3_hasher*>(global_hasher)->found_good_hash, 0, 1) == 0)
            {
                copy_good_nonce(hasher, reinterpret_cast<blake3_hasher*>(global_hasher));
            }
            atomicAdd(&reinterpret_cast<blake3_hasher*>(global_hasher)->hash_count, hasher->hash_count);
            return;
        }
    }
    atomicAdd(&reinterpret_cast<blake3_hasher*>(global_hasher)->hash_count, hasher->hash_count);
}
}

#endif //ALEPHIUM_ORIGINAL_BLAKE_H
