#!/usr/bin/env bash
# set -e

cd $(dirname $0)

while true
do
  time pipenv run python src/tower/run.py play --render --render-state
  sleep 5
done
