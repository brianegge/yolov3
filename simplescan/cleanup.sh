#!/bin/sh

set -o errexit
set -o xtrace

/usr/bin/find /srv/ftp/ -mindepth 1 -type f \( -name "*.dav" -o -name "*.idx" \) -delete
/usr/bin/find /srv/ftp/ -mindepth 2 -empty -type d -delete
