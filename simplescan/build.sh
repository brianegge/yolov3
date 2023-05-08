#!/bin/sh

cd $(dirname $0)

python -m compileall -q .
../bin/isort .
../bin/black .
