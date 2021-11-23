#ifndef ALEPHIUM_CONSTANTS_H
#define ALEPHIUM_CONSTANTS_H

#define group_nums 4
#define chain_nums 16
#define max_platform_num 10
#define max_gpu_num 1024
#define parallel_mining_works_per_gpu 4
#define max_worker_num (max_platform_num * max_gpu_num * parallel_mining_works_per_gpu)
#define mining_steps 5000

#define BLAKE3_BUF_CAP 384
#define BLAKE3_BUF_LEN 326

#endif // ALEPHIUM_CONSTANTS_H
