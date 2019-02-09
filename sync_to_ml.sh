#!/bin/sh

cd $(dirname $0)

rsync --exclude '*.pyc' \
  --exclude "data/" --exclude log --exclude obstacletower.app/ --exclude ml-agents/ \
  --exclude .git --exclude .idea/ --exclude log/ \
  --exclude '*.mp4' \
  -ruv ./ ml:workspace/tower $@
