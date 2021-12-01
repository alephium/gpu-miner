FROM nvidia/cuda:11.0-devel-ubuntu20.04 AS builder

WORKDIR /src

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install cmake tzdata python3-pip && \
    pip3 install conan

COPY ./ ./
RUN chmod +x make.sh && ./make.sh

FROM nvidia/cuda:11.0-base

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

COPY --from=builder /src/bin/gpu-miner /gpu-miner

USER root

ENTRYPOINT ["/gpu-miner"]
