#ifndef ALEPHIUM_LOG_H
#define ALEPHIUM_LOG_H

#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#define localtime_r(a, b) localtime_s(b, a)
#endif

#define LOG_TIMESTAMP(stream)                       \
{                                                   \
    time_t now = time(NULL);                        \
    struct tm time;                                 \
    localtime_r(&now, &time);                       \
    char str[24];                                   \
    strftime(str, 24, "%Y-%m-%d %H:%M:%S", &time);  \
    fprintf(stream, "%s | ", str);                  \
}

#define LOG_WITH_TS(stream, format, ...)            \
{                                                   \
    LOG_TIMESTAMP(stream);                          \
    fprintf(stream, format, ##__VA_ARGS__);         \
}

#define LOG_WITHOUT_TS(format, ...) fprintf(stdout, format, ##__VA_ARGS__);

#define LOG(format, ...) LOG_WITH_TS(stdout, format, ##__VA_ARGS__);

#define LOGERR(format, ...) LOG_WITH_TS(stderr, format, ##__VA_ARGS__);

#endif // ALEPHIUM_LOG_H
