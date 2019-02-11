#!/bin/sh
set -e

cd $(dirname $0)
#REMAIN_MEMORY=500
#echo Moving Memory
#find data/memory -name '*.gz' | grep pkl.gz | sort -r | sed -e "1,${REMAIN_MEMORY}d" | xargs rm -f
rsync -ruv data/memory/ ml:workspace/tower/data/memory

sleep 60
