#!/bin/sh

cd $(dirname $0)

../bin/isort .
../bin/black .
