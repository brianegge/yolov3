#!/bin/sh

set -o errexit

cd $(dirname $0)

. ../bin/activate
python -m compileall -q .
../bin/isort .
../bin/black .
python -m json.tool < excludes.json > /dev/null
