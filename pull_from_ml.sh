#!/bin/sh

cd $(dirname $0)

mkdir -p data/new_model

rsync --exclude '*.pyc' \
  -ruv ml:workspace/tower/data/model/ data/new_model
