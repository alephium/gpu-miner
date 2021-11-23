#ifndef ALEPHIUM_OPENCL_UTIL_H
#define ALEPHIUM_OPENCL_UTIL_H

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

#endif