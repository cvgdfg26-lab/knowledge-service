#!/bin/sh
# Execute on the CPU host. Keep this process alive (or manage it with your service manager).
set -eu
set -- ssh -CN \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes \
    -L 127.0.0.1:6006:127.0.0.1:6006 \
    -L 127.0.0.1:6008:127.0.0.1:6008 \
    -L 127.0.0.1:6009:127.0.0.1:6009 \
    -p "${GPU_SSH_PORT:-39560}"
if [ -n "${GPU_SSH_IDENTITY_FILE:-}" ]; then
    set -- "$@" -i "$GPU_SSH_IDENTITY_FILE" -o BatchMode=yes
fi
exec "$@" "${GPU_SSH_USER:-root}@${GPU_SSH_HOST:-connect.nmb1.seetacloud.com}"
