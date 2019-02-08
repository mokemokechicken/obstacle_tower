#!/usr/bin/env bash
set -e

cd $(dirname $0)

while true
do
  sh ./move_memory_to_ml.sh
  sh ./pull_from_ml.sh
done
