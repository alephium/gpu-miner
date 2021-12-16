#ifndef ALEPHIUM_INLINED_BLAKE_H
#define ALEPHIUM_INLINED_BLAKE_H

#include "blake3-common.hpp"

namespace inline_blake{

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
#define Z0A A
#define Z0B B
#define Z0C C
#define Z0D D
#define Z0E E
#define Z0F F
#define Z10 2
#define Z11 6
#define Z12 3
#define Z13 A
#define Z14 7
#define Z15 0
#define Z16 4
#define Z17 D
#define Z18 1
#define Z19 B
#define Z1A C
#define Z1B 5
#define Z1C 9
#define Z1D E
#define Z1E F
#define Z1F 8
#define Z20 3
#define Z21 4
#define Z22 A
#define Z23 C
#define Z24 D
#define Z25 2
#define Z26 7
#define Z27 E
#define Z28 6
#define Z29 5
#define Z2A 9
#define Z2B 0
#define Z2C B
#define Z2D F
#define Z2E 8
#define Z2F 1
#define Z30 A
#define Z31 7
#define Z32 C
#define Z33 9
#define Z34 E
#define Z35 3
#define Z36 D
#define Z37 F
#define Z38 4
#define Z39 0
#define Z3A B
#define Z3B 2
#define Z3C 5
#define Z3D 8
#define Z3E 1
#define Z3F 6
#define Z40 C
#define Z41 D
#define Z42 9
#define Z43 B
#define Z44 F
#define Z45 A
#define Z46 E
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
#define Z51 E
#define Z52 B
#define Z53 5
#define Z54 8
#define Z55 C
#define Z56 F
#define Z57 1
#define Z58 D
#define Z59 3
#define Z5A 0
#define Z5B A
#define Z5C 2
#define Z5D 6
#define Z5E 4
#define Z5F 7
#define Z60 B
#define Z61 F
#define Z62 5
#define Z63 0
#define Z64 1
#define Z65 9
#define Z66 8
#define Z67 6
#define Z68 E
#define Z69 A
#define Z6A 2
#define Z6B C
#define Z6C 3
#define Z6D 4
#define Z6E 7
#define Z6F D

INLINE __device__ uint32_t ROTR32(uint32_t w, uint32_t c)
{
    return (w >> c) | (w << (32 - c));
}

#define G(a, b, c, d, x, y)    \
    if (1)                     \
    {                          \
        a = a + b + x;         \
        d = ROTR32(d ^ a, 16); \
        c = c + d;             \
        b = ROTR32(b ^ c, 12); \
        a = a + b + y;         \
        d = ROTR32(d ^ a, 8);  \
        c = c + d;             \
        b = ROTR32(b ^ c, 7);  \
    }                          \
    else                       \
        ((void)0)

#define Mx(r, i) Mx_(Z##r##i)
#define Mx_(n) Mx__(n)
#define Mx__(n) M##n

#define ROUND(r)                               \
    if (1)                                     \
    {                                          \
        G(V0, V4, V8, VC, Mx(r, 0), Mx(r, 1)); \
        G(V1, V5, V9, VD, Mx(r, 2), Mx(r, 3)); \
        G(V2, V6, VA, VE, Mx(r, 4), Mx(r, 5)); \
        G(V3, V7, VB, VF, Mx(r, 6), Mx(r, 7)); \
        G(V0, V5, VA, VF, Mx(r, 8), Mx(r, 9)); \
        G(V1, V6, VB, VC, Mx(r, A), Mx(r, B)); \
        G(V2, V7, V8, VD, Mx(r, C), Mx(r, D)); \
        G(V3, V4, V9, VE, Mx(r, E), Mx(r, F)); \
    }                                          \
    else                                       \
        ((void)0)

#define COMPRESS_PRE \
    if (1)           \
    {                \
        V0 = H0;     \
        V1 = H1;     \
        V2 = H2;     \
        V3 = H3;     \
        V4 = H4;     \
        V5 = H5;     \
        V6 = H6;     \
        V7 = H7;     \
        V8 = IV_0;   \
        V9 = IV_1;   \
        VA = IV_2;   \
        VB = IV_3;   \
        VC = 0;      \
        VD = 0;      \
        VE = BLEN;   \
        VF = FLAGS;  \
                     \
        ROUND(0);    \
        ROUND(1);    \
        ROUND(2);    \
        ROUND(3);    \
        ROUND(4);    \
        ROUND(5);    \
        ROUND(6);    \
    }                \
    else             \
        ((void)0)

#define COMPRESS      \
    if (1)            \
    {                 \
        COMPRESS_PRE; \
        H0 = V0 ^ V8; \
        H1 = V1 ^ V9; \
        H2 = V2 ^ VA; \
        H3 = V3 ^ VB; \
        H4 = V4 ^ VC; \
        H5 = V5 ^ VD; \
        H6 = V6 ^ VE; \
        H7 = V7 ^ VF; \
    }                 \
    else              \
        ((void)0)

#define HASH_BLOCK(r, blen, flags) \
    if (1)                         \
    {                              \
        M0 = input[0x##r##0];          \
        M1 = input[0x##r##1];          \
        M2 = input[0x##r##2];          \
        M3 = input[0x##r##3];          \
        M4 = input[0x##r##4];          \
        M5 = input[0x##r##5];          \
        M6 = input[0x##r##6];          \
        M7 = input[0x##r##7];          \
        M8 = input[0x##r##8];          \
        M9 = input[0x##r##9];          \
        MA = input[0x##r##A];          \
        MB = input[0x##r##B];          \
        MC = input[0x##r##C];          \
        MD = input[0x##r##D];          \
        ME = input[0x##r##E];          \
        MF = input[0x##r##F];          \
        BLEN = (blen);             \
        FLAGS = (flags);           \
        COMPRESS;                  \
    }                              \
    else                           \
        ((void)0)

typedef struct
{
    uint8_t buf[BLAKE3_BUF_CAP];

    uint8_t hash[32]; // 64 bytes needed as hash will used as block words as well

    uint8_t target[32];
    uint32_t from_group;
    uint32_t to_group;

    uint32_t hash_count;
    int found_good_hash;
} blake3_hasher;

#define DOUBLE_HASH                             \
    if (1)                                      \
    {                                           \
        H1 = IV_1;                              \
        H0 = IV_0;                              \
        H2 = IV_2;                              \
        H3 = IV_3;                              \
        H4 = IV_4;                              \
        H5 = IV_5;                              \
        H6 = IV_6;                              \
        H7 = IV_7;                              \
        HASH_BLOCK(0, 64, CHUNK_START);         \
        HASH_BLOCK(1, 64, 0);                   \
        HASH_BLOCK(2, 64, 0);                   \
        HASH_BLOCK(3, 64, 0);                   \
        HASH_BLOCK(4, 64, 0);                   \
        HASH_BLOCK(5, 6, CHUNK_END | ROOT);     \
                                                \
        M0 = H0;                                \
        M1 = H1;                                \
        M2 = H2;                                \
        M3 = H3;                                \
        M4 = H4;                                \
        M5 = H5;                                \
        M6 = H6;                                \
        M7 = H7;                                \
        M8 = 0;                                 \
        M9 = 0;                                 \
        MA = 0;                                 \
        MB = 0;                                 \
        MC = 0;                                 \
        MD = 0;                                 \
        ME = 0;                                 \
        MF = 0;                                 \
        H0 = IV_0;                              \
        H1 = IV_1;                              \
        H2 = IV_2;                              \
        H3 = IV_3;                              \
        H4 = IV_4;                              \
        H5 = IV_5;                              \
        H6 = IV_6;                              \
        H7 = IV_7;                              \
        BLEN = 32;                              \
        FLAGS = CHUNK_START | CHUNK_END | ROOT; \
        COMPRESS;                               \
    }                                           \
    else                                        \
        ((void)0)

#define UPDATE_NONCE                                        \
    if (1)                                                  \
    {                                                       \
        if (atomicCAS(&reinterpret_cast<blake3_hasher*>(global_hasher)->found_good_hash, 0, 1) == 0) \
        {                                                   \
            uint32_t *nonce = (uint32_t *)reinterpret_cast<blake3_hasher*>(global_hasher)->buf;      \
            nonce[0] = input[0x00];                             \
            nonce[1] = input[0x01];                             \
            nonce[2] = input[0x02];                             \
            nonce[3] = input[0x03];                             \
            nonce[4] = input[0x04];                             \
            nonce[5] = input[0x05];                             \
            uint32_t *output = (uint32_t *)reinterpret_cast<blake3_hasher*>(global_hasher)->hash;    \
            output[0] = H0;                                 \
            output[1] = H1;                                 \
            output[2] = H2;                                 \
            output[3] = H3;                                 \
            output[4] = H4;                                 \
            output[5] = H5;                                 \
            output[6] = H6;                                 \
            output[7] = H7;                                 \
        }                                                   \
        atomicAdd(&reinterpret_cast<blake3_hasher*>(global_hasher)->hash_count, hash_count);         \
        return;                                             \
    }                                                       \
    else                                                    \
        ((void)0)

#define CHECK_INDEX                                                                         \
    if (1)                                                                                  \
    {                                                                                       \
        uint32_t big_index = (H7 & 0x0F000000) >> 24;                                       \
        if ((big_index / group_nums == from_group) && (big_index % group_nums == to_group)) \
        {                                                                                   \
            UPDATE_NONCE;                                                                   \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            goto cnt;                                                                       \
        }                                                                                   \
    }                                                                                       \
    else                                                                                    \
        ((void)0)

#define MASK0(n) (n & 0x000000FF)
#define MASK1(n) (n & 0x0000FF00)
#define MASK2(n) (n & 0x00FF0000)
#define MASK3(n) (n & 0xFF000000)
#define CHECK_TARGET(m, n)       \
    if (1)                       \
    {                            \
        m0 = MASK##n(H##m);      \
        m1 = MASK##n(target##m); \
        if (m0 > m1)             \
        {                        \
            goto cnt;            \
        }                        \
        else if (m0 < m1)        \
        {                        \
            CHECK_INDEX;         \
        }                        \
    }                            \
    else                         \
        ((void)0)

#define CHECK_POW           \
    if (1)                  \
    {                       \
        uint32_t m0, m1;    \
        CHECK_TARGET(0, 0); \
        CHECK_TARGET(0, 1); \
        CHECK_TARGET(0, 2); \
        CHECK_TARGET(0, 3); \
        CHECK_TARGET(1, 0); \
        CHECK_TARGET(1, 1); \
        CHECK_TARGET(1, 2); \
        CHECK_TARGET(1, 3); \
        CHECK_TARGET(2, 0); \
        CHECK_TARGET(2, 1); \
        CHECK_TARGET(2, 2); \
        CHECK_TARGET(2, 3); \
    }                       \
    else                    \
        ((void)0)

__global__ void blake3_hasher_mine(void *global_hasher)
{
    blake3_hasher hasher = *reinterpret_cast<blake3_hasher*>(global_hasher);
    uint32_t *input = (uint32_t *)hasher.buf;
    uint32_t *target = (uint32_t *)hasher.target;
    uint32_t target0 = target[0], target1 = target[1], target2 = target[2]; //, target3 = target[3], target4 = target[4], target5 = target[5], target6 = target[6], target7 = target[7];
    uint32_t from_group = hasher.from_group, to_group = hasher.to_group;
    uint32_t hash_count = 0;

    uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF; // message block
    uint32_t V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, VA, VB, VC, VD, VE, VF; // internal state
    uint32_t H0, H1, H2, H3, H4, H5, H6, H7;                                 // chain value
    uint32_t BLEN, FLAGS;                                                    // block len, flags

    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t *short_nonce = &input[0x00];
    *short_nonce = (*short_nonce) / stride * stride + tid;

    while (hash_count < mining_steps)
    {
        hash_count += 1;
        // printf("count: %u\n", hash_count);
        *short_nonce += stride;
        DOUBLE_HASH;
        CHECK_POW;
        cnt:;
    }
    atomicAdd(&reinterpret_cast<blake3_hasher*>(global_hasher)->hash_count, hash_count);
}

}

#endif //ALEPHIUM_INLINED_BLAKE_H
