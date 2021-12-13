#!/bin/bash

set -eu

API_KEY="0000000000000000000000000000000000000000000000000000000000000000"

if ! command -v jq &> /dev/null
then
    echo "Installing jq"
    sudo apt install -y jq
fi

node=$(curl --silent -X 'GET' "http://${2:-127.0.0.1}:12973/infos/self-clique" -H "X-API-KEY: $API_KEY")
if [ -z "$node" ]; then
    echo "Your full node is not running"
    exit 1
fi

synced=$(echo $node | jq '.synced')
if [ -z "$synced" ] || [ "$synced" != "true" ]; then
    echo "Your full node is not synced or API key is not correct: $node"
    exit 1
fi

addresses=$(curl --silent -X 'GET' "http://${2:-127.0.0.1}:12973/miners/addresses" -H "X-API-KEY: $API_KEY" | jq '.addresses')
if [ -z "$addresses" ]; then
    echo "Miner addresses are not set"
    exit 1
fi

SCRIPT_DIR=`dirname "$BASH_SOURCE"`

echo "Launching the miner and restart automatically if it crashes"
until $SCRIPT_DIR/bin/gpu-miner $*; do
    echo "Miner crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
