#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "messages.h"

#define INLINE __forceinline__
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
    do                         \
    {                          \
        a = a + b + x;         \
        d = ROTR32(d ^ a, 16); \
        c = c + d;             \
        b = ROTR32(b ^ c, 12); \
        a = a + b + y;         \
        d = ROTR32(d ^ a, 8);  \
        c = c + d;             \
        b = ROTR32(b ^ c, 7);  \
    } while (0)

#define Mx(r, i) Mx_(Z##r##i)
#define Mx_(n) Mx__(n)
#define Mx__(n) M##n

#define ROUND(r)                               \
    do                                         \
    {                                          \
        G(V0, V4, V8, VC, Mx(r, 0), Mx(r, 1)); \
        G(V1, V5, V9, VD, Mx(r, 2), Mx(r, 3)); \
        G(V2, V6, VA, VE, Mx(r, 4), Mx(r, 5)); \
        G(V3, V7, VB, VF, Mx(r, 6), Mx(r, 7)); \
        G(V0, V5, VA, VF, Mx(r, 8), Mx(r, 9)); \
        G(V1, V6, VB, VC, Mx(r, A), Mx(r, B)); \
        G(V2, V7, V8, VD, Mx(r, C), Mx(r, D)); \
        G(V3, V4, V9, VE, Mx(r, E), Mx(r, F)); \
    } while (0)

#define COMPRESS_PRE \
    do               \
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
    } while (0)

#define COMPRESS      \
    do                \
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
    } while (0)

#define HASH_BLOCK(r, blen, flags) \
    do                             \
    {                              \
        M0 = input##r##0;          \
        M1 = input##r##1;          \
        M2 = input##r##2;          \
        M3 = input##r##3;          \
        M4 = input##r##4;          \
        M5 = input##r##5;          \
        M6 = input##r##6;          \
        M7 = input##r##7;          \
        M8 = input##r##8;          \
        M9 = input##r##9;          \
        MA = input##r##A;          \
        MB = input##r##B;          \
        MC = input##r##C;          \
        MD = input##r##D;          \
        ME = input##r##E;          \
        MF = input##r##F;          \
        BLEN = (blen);             \
        FLAGS = (flags);           \
        COMPRESS;                  \
    } while (0)

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

#define DOUBLE_HASH                                  \
    do                                               \
    {                                                \
        H1 = IV_1;                                   \
        H0 = IV_0;                                   \
        H2 = IV_2;                                   \
        H3 = IV_3;                                   \
        H4 = IV_4;                                   \
        H5 = IV_5;                                   \
        H6 = IV_6;                                   \
        H7 = IV_7;                                   \
        HASH_BLOCK(0, 64, CHUNK_START);              \
        HASH_BLOCK(1, 64, 0);                        \
        HASH_BLOCK(2, 64, 0);                        \
        HASH_BLOCK(3, 64, 0);                        \
        HASH_BLOCK(4, 64, 0);                        \
        HASH_BLOCK(5, 6, CHUNK_END | ROOT);          \
                                                     \
        M0 = H0;                                     \
        M1 = H1;                                     \
        M2 = H2;                                     \
        M3 = H3;                                     \
        M4 = H4;                                     \
        M5 = H5;                                     \
        M6 = H6;                                     \
        M7 = H7;                                     \
        M8 = 0;                                      \
        M9 = 0;                                      \
        MA = 0;                                      \
        MB = 0;                                      \
        MC = 0;                                      \
        MD = 0;                                      \
        ME = 0;                                      \
        MF = 0;                                      \
        H0 = IV_0;                                   \
        H1 = IV_1;                                   \
        H2 = IV_2;                                   \
        H3 = IV_3;                                   \
        H4 = IV_4;                                   \
        H5 = IV_5;                                   \
        H6 = IV_6;                                   \
        H7 = IV_7;                                   \
        BLEN = 32;                                   \
        FLAGS = CHUNK_START | CHUNK_END | ROOT;      \
        COMPRESS;                                    \
    } while (0)

#define UPDATE_NONCE                                        \
    do                                                      \
    {                                                       \
        if (atomicCAS(&hasher->found_good_hash, 0, 1) == 0) \
        {                                                   \
            uint32_t *nonce = (uint32_t *)hasher->buf;      \
            nonce[0] = input00;                             \
            nonce[1] = input01;                             \
            nonce[2] = input02;                             \
            nonce[3] = input03;                             \
            nonce[4] = input04;                             \
            nonce[5] = input05;                             \
            uint32_t *output = (uint32_t *)hasher->hash;    \
            output[0] = H0;                                 \
            output[1] = H1;                                 \
            output[2] = H2;                                 \
            output[3] = H3;                                 \
            output[4] = H4;                                 \
            output[5] = H5;                                 \
            output[6] = H6;                                 \
            output[7] = H7;                                 \
        }                                                   \
        atomicAdd(&hasher->hash_count, hash_count);         \
        return;                                             \
    } while (0)

#define CHECK_INDEX                                                                         \
    do                                                                                      \
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
    } while (0)

#define MASK0(n) (n & 0x000000FF)
#define MASK1(n) (n & 0x0000FF00)
#define MASK2(n) (n & 0x00FF0000)
#define MASK3(n) (n & 0xFF000000)
#define CHECK_TARGET(m, n)       \
    do                           \
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
    } while (0)

#define CHECK_POW           \
    do                      \
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
    } while (0)

__global__ void blake3_hasher_mine(blake3_hasher *hasher)
{
    uint32_t *input = (uint32_t *)hasher->buf;
    uint32_t input00 = input[0x00], input01 = input[0x01], input02 = input[0x02], input03 = input[0x03], input04 = input[0x04], input05 = input[0x05], input06 = input[0x06], input07 = input[0x07], input08 = input[0x08], input09 = input[0x09], input0A = input[0x0A], input0B = input[0x0B], input0C = input[0x0C], input0D = input[0x0D], input0E = input[0x0E], input0F = input[0x0F];
    uint32_t input10 = input[0x10], input11 = input[0x11], input12 = input[0x12], input13 = input[0x13], input14 = input[0x14], input15 = input[0x15], input16 = input[0x16], input17 = input[0x17], input18 = input[0x18], input19 = input[0x19], input1A = input[0x1A], input1B = input[0x1B], input1C = input[0x1C], input1D = input[0x1D], input1E = input[0x1E], input1F = input[0x1F];
    uint32_t input20 = input[0x20], input21 = input[0x21], input22 = input[0x22], input23 = input[0x23], input24 = input[0x24], input25 = input[0x25], input26 = input[0x26], input27 = input[0x27], input28 = input[0x28], input29 = input[0x29], input2A = input[0x2A], input2B = input[0x2B], input2C = input[0x2C], input2D = input[0x2D], input2E = input[0x2E], input2F = input[0x2F];
    uint32_t input30 = input[0x30], input31 = input[0x31], input32 = input[0x32], input33 = input[0x33], input34 = input[0x34], input35 = input[0x35], input36 = input[0x36], input37 = input[0x37], input38 = input[0x38], input39 = input[0x39], input3A = input[0x3A], input3B = input[0x3B], input3C = input[0x3C], input3D = input[0x3D], input3E = input[0x3E], input3F = input[0x3F];
    uint32_t input40 = input[0x40], input41 = input[0x41], input42 = input[0x42], input43 = input[0x43], input44 = input[0x44], input45 = input[0x45], input46 = input[0x46], input47 = input[0x47], input48 = input[0x48], input49 = input[0x49], input4A = input[0x4A], input4B = input[0x4B], input4C = input[0x4C], input4D = input[0x4D], input4E = input[0x4E], input4F = input[0x4F];
    uint32_t input50 = input[0x50], input51 = input[0x51], input52 = input[0x52], input53 = input[0x53], input54 = input[0x54], input55 = input[0x55], input56 = input[0x56], input57 = input[0x57], input58 = input[0x58], input59 = input[0x59], input5A = input[0x5A], input5B = input[0x5B], input5C = input[0x5C], input5D = input[0x5D], input5E = input[0x5E], input5F = input[0x5F];
    uint32_t *target = (uint32_t *)hasher->target;
    uint32_t target0 = target[0], target1 = target[1], target2 = target[2]; //, target3 = target[3], target4 = target[4], target5 = target[5], target6 = target[6], target7 = target[7];
    uint32_t from_group = hasher->from_group, to_group = hasher->to_group;
    uint32_t hash_count = 0;

    uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF; // message block
    uint32_t V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, VA, VB, VC, VD, VE, VF; // internal state
    uint32_t H0, H1, H2, H3, H4, H5, H6, H7;                                 // chain value
    uint32_t BLEN, FLAGS;                                                    // block len, flags

    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t *short_nonce = &input00;
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
    atomicAdd(&hasher->hash_count, hash_count);
}

#ifdef BLAKE3_TEST
int main()
{
    blob_t target;
    hex_to_bytes("00009fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", &target);

    blake3_hasher *hasher;
    blake3_hasher *device_hasher;
    TRY(cudaMallocHost(&hasher, sizeof(blake3_hasher)));
    TRY(cudaMalloc(&device_hasher, sizeof(blake3_hasher)));

    bzero(hasher->buf, BLAKE3_BUF_CAP);
    memcpy(hasher->target, target.blob, target.len);
    hasher->from_group = 0;
    hasher->to_group = 3;

    cudaStream_t stream;
    TRY(cudaStreamCreate(&stream));
    TRY(cudaMemcpyAsync(device_hasher, hasher, sizeof(blake3_hasher), cudaMemcpyHostToDevice, stream));
    blake3_hasher_mine<<<32, 32, 0, stream>>>(device_hasher);
    TRY(cudaStreamSynchronize(stream));

    TRY(cudaMemcpy(hasher, device_hasher, sizeof(blake3_hasher), cudaMemcpyDeviceToHost));
    char *hash_string1 = bytes_to_hex(hasher->hash, 32);
    printf("good: %d\n", hasher->found_good_hash);
    printf("nonce: %d\n", hasher->buf[0]);
    printf("count: %d\n", hasher->hash_count);
    printf("%s\n", hash_string1); // 0003119e5bf02115e1c8496008fbbcec4884e0be7f9dc372cd4316a51d065283
}
#endif // BLAKE3_TEST