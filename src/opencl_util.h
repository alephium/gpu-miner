#ifndef ALEPHIUM_OPENCL_UTIL_H
#define ALEPHIUM_OPENCL_UTIL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define TRY(x)                                                                                  \
    {                                                                                           \
        cl_int err = (x);                                                                       \
        if (err != CL_SUCCESS)                                                                  \
        {                                                                                       \
            printf("opencl error %d calling '%s' (%s line %d)\n", err, #x, __FILE__, __LINE__); \
        }                                                                                       \
    }

#define CHECK(x)                                                                                \
    {                                                                                           \
        x;                                                                                    \
        if (err != CL_SUCCESS)                                                                  \
        {                                                                                       \
            printf("opencl error %d calling '%s' (%s line %d)\n", err, #x, __FILE__, __LINE__); \
        }                                                                                       \
    }

char *load_kernel_source(const char *source_file)
{
    // Load the kernel source code into the array source_str
    FILE *fp = fopen(source_file, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file '%s'\n", source_file);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);

    char *source_str = (char *)malloc(file_size + 1);
    source_str[file_size] = '\0';
    rewind(fp);
    fread(source_str, sizeof(char), file_size, fp);
    fclose(fp);
    return source_str;
}

#endif