FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 AS builder

WORKDIR /src

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y libuv1-dev

COPY . /src
RUN make gpu

FROM ubuntu:20.04

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y libuv1
RUN rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/bin/gpu-miner /gpu-miner

USER nobody

ENTRYPOINT ["/gpu-miner"]
