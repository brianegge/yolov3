#!/bin/sh

cd $(dirname $0)

. ../bin/activate
python -m compileall -q .
../bin/isort .
../bin/black .
