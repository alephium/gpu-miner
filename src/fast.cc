#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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

#define ROTR32(w, c) (((w) >> (c)) | ((w) << (32 - (c))))

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
        M0 = input[0x##r##0];      \
        M1 = input[0x##r##1];      \
        M2 = input[0x##r##2];      \
        M3 = input[0x##r##3];      \
        M4 = input[0x##r##4];      \
        M5 = input[0x##r##5];      \
        M6 = input[0x##r##6];      \
        M7 = input[0x##r##7];      \
        M8 = input[0x##r##8];      \
        M9 = input[0x##r##9];      \
        MA = input[0x##r##A];      \
        MB = input[0x##r##B];      \
        MC = input[0x##r##C];      \
        MD = input[0x##r##D];      \
        ME = input[0x##r##E];      \
        MF = input[0x##r##F];      \
        BLEN = (blen);             \
        FLAGS = (flags);           \
        COMPRESS;                  \
    } while (0)

// input.len == 326, i.e. 326 / 4 = 81.5 words, 326 / 64 = 5.09375 message blocks
void blake3_double_hash(uint32_t *input, uint32_t *output)
{
    uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF; // message block
    uint32_t V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, VA, VB, VC, VD, VE, VF; // internal state
    uint32_t H0, H1, H2, H3, H4, H5, H6, H7;                                 // chain value
    uint32_t BLEN, FLAGS;                                                    // block len, flags

    H0 = IV_0;
    H1 = IV_1;
    H2 = IV_2;
    H3 = IV_3;
    H4 = IV_4;
    H5 = IV_5;
    H6 = IV_6;
    H7 = IV_7;
    HASH_BLOCK(0, 0, CHUNK_START | CHUNK_END | ROOT);

    output[0] = H0;
    output[1] = H1;
    output[2] = H2;
    output[3] = H3;
    output[4] = H4;
    output[5] = H5;
    output[6] = H6;
    output[7] = H7;
}

// typedef struct
// {
//     uint8_t buf[BLAKE3_BUF_CAP];

//     uint32_t cv[8];

//     uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

//     uint8_t target[32];
//     uint32_t from_group;
//     uint32_t to_group;

//     uint32_t hash_count;
//     int found_good_hash;
// } blake3_hasher;

// void blake3_hasher_mine(blake3_hasher *hasher)
// {

// }

#include "messages.h"

int main()
{
    uint32_t input[16] = {0};
    uint32_t output[8];

    print_hex("input", (uint8_t *)input, 64);
    blake3_double_hash(input, output);
    print_hex("output", (uint8_t *)output, 32);
}