#!/bin/bash

set -eu

if ! command -v jq &> /dev/null
then 
    echo "Installing jq"
    sudo apt install -y jq
fi

if ! node=$(curl --silent -X 'GET' 'http://127.0.0.1:12973/infos/self-clique'); then
    echo "Your full node is not running"
    exit 1
fi

synced=$(echo $node | jq '.synced')
if [ -z "$synced" ] || [ "$synced" != "true" ]; then
    echo "Your full node is not synced"
    exit 1
fi

addresses=$(curl --silent -X 'GET' 'http://127.0.0.1:12973/miners/addresses' | jq '.addresses')
if [ -z "$addresses" ]; then
    echo "Miner addresses are not set"
    exit 1
fi

SCRIPT_DIR=`dirname "$BASH_SOURCE"`

echo "Launching the miner and restart automatically if it crashes"
until $SCRIPT_DIR/bin/gpu-miner; do
    echo "Miner crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
