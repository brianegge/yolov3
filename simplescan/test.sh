#!/bin/bash
set -e
podman run --rm -v "$(pwd)":/app -w /app simplescan-test python -m compileall -q .
podman run --rm -v "$(pwd)":/app -w /app simplescan-test pytest -v utils_test.py
