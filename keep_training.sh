#!/usr/bin/env bash
set -e

cd $(dirname $0)

while true
do
  time pipenv run python src/tower/run.py train
done
